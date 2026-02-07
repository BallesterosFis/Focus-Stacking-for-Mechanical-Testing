import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
import time
import sys
import cv2
import pywt 
from datetime import datetime

try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except ImportError:
        print("Critical Error: MoviePy is not installed.")
        sys.exit(1)

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("Warning: ReportLab is not installed. PDF report generation will be unavailable.")
    REPORTLAB_AVAILABLE = False

WAVELET_TYPE = "haar"
DOF_THRESHOLD = 0.8         
FWHM_THRESHOLD = 0.5      
ROI_FIXED_SIZE = 1024       
GIF_FPS = 5

# REGION OF INTEREST (ROI) SELECTION
def select_square_roi(image):
    if image is None:
        raise ValueError("Error: Null image provided for ROI selection.")

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    display_scale = 1.0
    h_orig, w_orig = image.shape[:2]
    
    effective_side = min(ROI_FIXED_SIZE, h_orig, w_orig)
    if effective_side < ROI_FIXED_SIZE:
        print(f"Advertencia: La imagen ({w_orig}x{h_orig}) es menor que el ROI solicitado ({ROI_FIXED_SIZE}). Se ajustará a {effective_side}.")


    if h_orig > 900 or w_orig > 1600: 
        display_scale = min(900 / h_orig, 1600 / w_orig)
        image_display = cv2.resize(image_rgb, (0,0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    else:
        image_display = image_rgb

    plt.ioff()
    fig, ax = plt.subplots(figsize=(image_display.shape[1]/100, image_display.shape[0]/100), dpi=100) 
    ax.imshow(image_display)
    ax.set_title(f"ROI Selection: Click the CENTER of the {effective_side}x{effective_side} box")
    plt.axis('off') 
    
    print("\n--- ROI SELECTION INSTRUCTIONS ---")
    print(f"A window will open to define the Region of Interest.")
    print(f"Target Size: {effective_side} x {effective_side} pixels.")
    print("1. LEFT CLICK: Select the CENTER point of the region.")
    print("---------------------------------------------")

    pts = plt.ginput(1, timeout=-1) 
    
    # Default fallback if selection fails (Center of image)
    if len(pts) < 1:
        print("Warning: No selection made. Using image center.")
        center_x_disp = (w_orig * display_scale) / 2
        center_y_disp = (h_orig * display_scale) / 2
    else:
        center_x_disp, center_y_disp = pts[0]

    # Convert display coordinates to original coordinates
    center_x_orig = int(center_x_disp / display_scale)
    center_y_orig = int(center_y_disp / display_scale)

    # Calculate Top-Left based on Center
    half_side = effective_side // 2
    x_min = center_x_orig - half_side
    y_min = center_y_orig - half_side

    # Boundary Checks (Clamp to image limits)
    if x_min < 0: x_min = 0
    if y_min < 0: y_min = 0
    if x_min + effective_side > w_orig: x_min = w_orig - effective_side
    if y_min + effective_side > h_orig: y_min = h_orig - effective_side
    
    x_final = int(x_min)
    y_final = int(y_min)
    
    # Visual confirmation
    rect_disp_x = x_final * display_scale
    rect_disp_y = y_final * display_scale
    rect_disp_side = effective_side * display_scale

    rect = patches.Rectangle((rect_disp_x, rect_disp_y), rect_disp_side, rect_disp_side, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"ROI Fixed ({effective_side}x{effective_side}) Selected. Closing...")
    plt.draw()
    plt.pause(1.5)
    plt.close(fig)

    return x_final, y_final, effective_side

# IMAGE QUALITY METRICS
def laplacian_variance(img):
    """Calculates the variance of the Laplacian operator (Edge Detection)."""
    return cv2.Laplacian(img, cv2.CV_64F).var()

def tenengrad(img):
    """Calculates the squared gradient magnitude (Sobel)."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx**2 + gy**2)

def wavelet_energy(img, wavelet="haar", level=1):
    """Calculates the energy of high-frequency coefficients using Discrete Wavelet Transform."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    cH, cV, cD = coeffs[-1] 
    return np.mean(cH**2 + cV**2 + cD**2)


# PDF REPORT GENERATION
def create_summary_plot(z_indexes, lap_vals, ten_vals, wav_vals, dof_results):
    """Generates the summary figure for the PDF report."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Focus Analysis Summary', fontsize=14, fontweight='bold')
    
    # Normalization
    lap_norm = lap_vals / (np.max(lap_vals) + 1e-6)
    ten_norm = ten_vals / (np.max(ten_vals) + 1e-6)
    wav_norm = wav_vals / (np.max(wav_vals) + 1e-6)
    
    # 1. Normalized Metrics & Regions
    ax1 = axes[0, 0]
    ax1.plot(z_indexes, lap_norm, 'b-', label='Laplacian', linewidth=1.5)
    ax1.plot(z_indexes, ten_norm, 'g-', label='Tenengrad', linewidth=1.5)
    ax1.plot(z_indexes, wav_norm, 'm-', label=f'Wavelet', linewidth=2)
    
    # Linea de referencia del 50%
    ax1.axhline(y=FWHM_THRESHOLD, color='r', linestyle='--', alpha=0.5, label='50% Height')
    ax1.axhline(y=DOF_THRESHOLD, color='k', linestyle=':', alpha=0.5, label=f'{int(DOF_THRESHOLD*100)}% Height')

    ax1.set_title('Normalized Metrics & DoF Ranges')
    ax1.set_xlabel('Z-Index')
    ax1.set_ylabel('Normalized Magnitude')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Visualization of DoF regions
    for metric_name, dof_info in dof_results.items():
        if dof_info['dof_start_80'] is not None:
            color = {'Laplacian': 'blue', 'Tenengrad': 'green', f'Wavelet ({WAVELET_TYPE})': 'purple'}.get(metric_name, 'gray')
            ax1.axvspan(dof_info['dof_start_80'], dof_info['dof_end_80'], alpha=0.1, color=color)
    
    # 2. DoF Width Comparison
    ax2 = axes[0, 1]
    metric_names = list(dof_results.keys())
    widths_80 = []
    widths_50 = []
    
    for metric_name in metric_names:
        dof_info = dof_results[metric_name]
        if dof_info['dof_start_80'] is not None:
            widths_80.append(dof_info['dof_end_80'] - dof_info['dof_start_80'] + 1)
        else:
            widths_80.append(0)
        if dof_info['dof_start_50'] is not None:
            widths_50.append(dof_info['dof_end_50'] - dof_info['dof_start_50'] + 1)
        else:
            widths_50.append(0)
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, widths_80, width, label='Strict (80%)', color='#1f77b4')
    rects2 = ax2.bar(x + width/2, widths_50, width, label='FWHM (50%)', color='#ff7f0e')
    
    ax2.set_title('DoF Width Comparison')
    ax2.set_ylabel('Number of Images')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name[:10] for name in metric_names], rotation=45)
    ax2.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax2.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    autolabel(rects1)
    autolabel(rects2)

    # 3. Absolute Values
    ax3 = axes[1, 0]
    ax3.plot(z_indexes, lap_vals, 'b-o', markersize=3, label='Laplacian', alpha=0.7)
    ax3.plot(z_indexes, ten_vals, 'g-s', markersize=3, label='Tenengrad', alpha=0.7)
    ax3.plot(z_indexes, wav_vals, 'm-^', markersize=3, label=f'Wavelet', alpha=0.7)
    ax3.set_title('Absolute Values')
    ax3.set_xlabel('Z-Index')
    ax3.set_ylabel('Metric Value')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Maximum Focus Points
    ax4 = axes[1, 1]
    peak_data = []
    for metric_name in metric_names:
        dof_info = dof_results[metric_name]
        if dof_info['peak_index'] is not None:
            peak_data.append({
                'metric': metric_name,
                'z_index': dof_info['peak_index'],
                'value': dof_info['peak_value']
            })
    
    if peak_data:
        peak_indices = [d['z_index'] for d in peak_data]
        peak_values = [d['value'] for d in peak_data]
        metric_labels = [d['metric'] for d in peak_data]
        
        colors_scatter = ['#1f77b4', '#2ca02c', '#9467bd'][:len(peak_data)]
        ax4.scatter(peak_indices, peak_values, s=100, c=colors_scatter, alpha=0.7)
        
        for i, (idx, val, label) in enumerate(zip(peak_indices, peak_values, metric_labels)):
            ax4.annotate(f'{label.split()[0]}\nZ={idx}', 
                        (idx, val),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax4.set_title('Point of Maximum Focus')
    ax4.set_xlabel('Z-Index')
    ax4.set_ylabel('Peak Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_pdf_report(folder_name, analysis_results, roi_info, dof_results, gif_path=None):
    """Generates a technical report in PDF format."""
    if not REPORTLAB_AVAILABLE:
        return None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, folder_name)
    graphs_dir = os.path.join(save_path, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(save_path, f"Analysis_Report_{timestamp}.pdf")
    
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=72, bottomMargin=72,
        title=f"Analysis Report - {folder_name}"
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=24,
        alignment=TA_CENTER,
        textColor=colors.black,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'Heading1Style',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.black,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    story = []
    
    # ===== COVER =====
    story.append(Spacer(1, 80))
    story.append(Paragraph("FOCUS ANALYSIS REPORT (Z-STACK)", title_style))
    story.append(Spacer(1, 20))
    
    portada_data = [
        ["Dataset Identifier:", folder_name],
        ["Processing Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Processed Images:", str(analysis_results.get('num_images', 'N/A'))],
        ["ROI Dimension:", f"{roi_info[2]} x {roi_info[2]} pixels"],
        ["ROI Origin:", f"X={roi_info[0]}, Y={roi_info[1]}"],
    ]
    
    portada_table = Table(portada_data, colWidths=[2.5*inch, 3*inch])
    portada_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    story.append(portada_table)
    story.append(PageBreak())
    
    # ===== RESULTS =====
    story.append(Paragraph("SUMMARY OF RESULTS", heading1_style))
    
    summary_text = f"""
    The analysis includes two focus thresholds:
    1. <b>Strict Focus ({int(DOF_THRESHOLD*100)}% Peak):</b> Indicates the region of sharpest focus.
    2. <b>FWHM ({int(FWHM_THRESHOLD*100)}% Peak):</b> Full Width at Half Maximum, indicating the broad focus range.
    """
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 15))
    
    # DoF Table
    dof_headers = ["Metric", "Peak Z", "Start(80%)", "End(80%)", "Width(80%)", "Start(50%)", "End(50%)", "Width(50%)"]
    dof_table_data = [dof_headers]
    
    for metric_name, dof_info in dof_results.items():
        peak_z = str(dof_info['peak_index']) if dof_info['peak_index'] is not None else "N/A"
        
        if dof_info['dof_start_80'] is not None:
            w80 = dof_info['dof_end_80'] - dof_info['dof_start_80'] + 1
            s80, e80 = str(dof_info['dof_start_80']), str(dof_info['dof_end_80'])
            w80_str = str(w80)
        else:
            s80, e80, w80_str = "N/A", "N/A", "N/A"

        if dof_info['dof_start_50'] is not None:
            w50 = dof_info['dof_end_50'] - dof_info['dof_start_50'] + 1
            s50, e50 = str(dof_info['dof_start_50']), str(dof_info['dof_end_50'])
            w50_str = str(w50)
        else:
            s50, e50, w50_str = "N/A", "N/A", "N/A"

        row = [metric_name, peak_z, s80, e80, w80_str, s50, e50, w50_str]
        dof_table_data.append(row)
    
    col_widths = [1.2*inch] + [0.7*inch]*7
    dof_table = Table(dof_table_data, colWidths=col_widths)
    dof_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (5, 1), (7, -1), colors.whitesmoke),
    ]))
    
    story.append(dof_table)
    story.append(Spacer(1, 20))
    
    # ===== PLOTS =====
    story.append(Paragraph("METRIC VISUALIZATION", heading1_style))
    
    z_indexes = np.arange(analysis_results.get('num_images', 0))
    lap_vals = analysis_results.get('metrics', {}).get('Laplacian', [])
    ten_vals = analysis_results.get('metrics', {}).get('Tenengrad', [])
    wav_vals = analysis_results.get('metrics', {}).get(f'Wavelet ({WAVELET_TYPE})', [])
    
    if len(lap_vals) > 0:
        summary_fig = create_summary_plot(z_indexes, lap_vals, ten_vals, wav_vals, dof_results)
        summary_path = os.path.join(graphs_dir, "summary_plot.png")
        summary_fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(summary_fig)
        
        img = ReportLabImage(summary_path, width=7*inch, height=5.5*inch)
        story.append(img)
        story.append(Paragraph("Figure 1: Comparison of metrics with 80% (Strict) and 50% (FWHM) thresholds.", normal_style))
    
    # ===== PARAMETERS =====
    story.append(PageBreak())
    story.append(Paragraph("ANALYSIS PARAMETERS", heading1_style))
    
    params_data = [
        ["Parameter", "Configuration", "Description"],
        ["Wavelet Type", WAVELET_TYPE, "Base for wavelet transform"],
        ["ROI Size", f"{ROI_FIXED_SIZE}x{ROI_FIXED_SIZE}", "Fixed dimensions for analysis window"],
        ["DoF Strict Threshold", f"{DOF_THRESHOLD}", "Relative threshold (0-1) for sharp focus"],
        ["DoF FWHM Threshold", f"{FWHM_THRESHOLD}", "Relative threshold (0-1) for half-max width"],
        ["GIF FPS", str(GIF_FPS), "Animation playback speed"],
    ]
    
    params_table = Table(params_data)
    params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    
    story.append(params_table)
    
    try:
        doc.build(story)
        return pdf_filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

# ANALYSIS LOGIC
def analyze_stack(folder_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, folder_name)
    
    if not os.path.exists(path):
        print(f"Error: The directory '{path}' does not exist.")
        return None
    
    files = sorted([f for f in os.listdir(path) if f.lower().endswith((".jpeg", ".jpg", ".png", ".tiff", ".tif"))])
    
    if not files:
        print("Error: No valid image files found in the directory.")
        return None
    
    plt.close('all')
    
    # Load reference image
    ref_img_path = os.path.join(path, files[len(files)//2])
    ref_img = cv2.imread(ref_img_path)
    
    if ref_img is None:
        print(f"Critical Error: Failed to read reference image: {ref_img_path}")
        return None
    
    # ROI Selection
    try:
        x_roi, y_roi, side_roi = select_square_roi(ref_img)
        print(f"ROI Established: Origin({x_roi}, {y_roi}), Side={side_roi}")
    except Exception as e:
        print(f"Exception during ROI selection: {e}")
        return None
    
    lap_vals, ten_vals, wav_vals = [], [], []
    gif_frames = []
    
    print(f"Processing {len(files)} images...")
    
    for idx, f in enumerate(files):
        img_full = cv2.imread(os.path.join(path, f))
        if img_full is None:
            continue
        
        gray = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
        roi_img = gray[y_roi:y_roi+side_roi, x_roi:x_roi+side_roi]
        
        # Metric Calculation
        lap_vals.append(laplacian_variance(roi_img))
        ten_vals.append(tenengrad(roi_img))
        wav_vals.append(wavelet_energy(roi_img, WAVELET_TYPE))
        
        # GIF Preparation
        img_for_gif = img_full.copy()
        cv2.rectangle(img_for_gif, (x_roi, y_roi), (x_roi + side_roi, y_roi + side_roi), (0, 0, 255), 2)
        
        current_wav_val = wav_vals[-1] if wav_vals else 0
        text = f"ID:{idx} | Wav:{current_wav_val:.2f}"
        cv2.putText(img_for_gif, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Resize GIF
        gif_display_h = 480
        gif_display_w = int(img_full.shape[1] * (gif_display_h / img_full.shape[0]))
        img_for_gif_resized = cv2.resize(img_for_gif, (gif_display_w, gif_display_h), interpolation=cv2.INTER_AREA)
        gif_frames.append(img_for_gif_resized)
        
        if idx % 10 == 0:
            print(f"Processing... {idx}/{len(files)}")
            
    print("Metric calculation finished.")
    
    lap_vals = np.array(lap_vals)
    ten_vals = np.array(ten_vals)
    wav_vals = np.array(wav_vals)
    z_indexes = np.arange(len(files))
    
    def calculate_dof_range(metric_values, threshold):
        if len(metric_values) == 0:
            return None, None, None, None
        
        peak_idx = np.argmax(metric_values)
        peak_val = metric_values[peak_idx]
        
        above_threshold_indices = np.where(metric_values >= (peak_val * threshold))[0]
        
        if len(above_threshold_indices) == 0:
            return None, None, peak_idx, peak_val
        
        start_dof_idx = above_threshold_indices[0]
        end_dof_idx = above_threshold_indices[-1]
        
        return start_dof_idx, end_dof_idx, peak_idx, peak_val
    
    dof_results = {}
    metrics_map = {
        'Laplacian': lap_vals,
        'Tenengrad': ten_vals,
        f'Wavelet ({WAVELET_TYPE})': wav_vals
    }

    for name, vals in metrics_map.items():
        s80, e80, p_idx, p_val = calculate_dof_range(vals, DOF_THRESHOLD)
        s50, e50, _, _ = calculate_dof_range(vals, FWHM_THRESHOLD)
        
        dof_results[name] = {
            'peak_index': p_idx,
            'peak_value': p_val,
            'dof_start_80': s80,
            'dof_end_80': e80,
            'dof_start_50': s50,
            'dof_end_50': e50
        }
    
    analysis_results = {
        'num_images': len(files),
        'metrics': {
            'Laplacian': lap_vals.tolist(),
            'Tenengrad': ten_vals.tolist(),
            f'Wavelet ({WAVELET_TYPE})': wav_vals.tolist()
        }
    }
    
    roi_info = (x_roi, y_roi, side_roi)
    
    gif_path = None
    if gif_frames:
        gif_path = os.path.join(path, f"animation_roi.gif")
        try:
            clip = ImageSequenceClip(gif_frames, fps=GIF_FPS)
            clip.write_gif(gif_path, fps=GIF_FPS, loop=0, verbose=False, logger=None)
            print(f"GIF generated: {gif_path}")
        except Exception as e:
            print(f"Error generating GIF: {e}")

    pdf_path = None
    if REPORTLAB_AVAILABLE:
        print("Generating PDF report...")
        pdf_path = generate_pdf_report(folder_name, analysis_results, roi_info, dof_results, gif_path)
        if pdf_path:
            print(f"PDF report saved to: {pdf_path}")
    
    results_json = {
        'metadata': {
            'folder': folder_name,
            'timestamp': datetime.now().isoformat(),
            'wavelet_type': WAVELET_TYPE,
            'roi_size': ROI_FIXED_SIZE,
            'dof_threshold_strict': DOF_THRESHOLD,
            'dof_threshold_fwhm': FWHM_THRESHOLD
        },
        'roi': roi_info,
        'results_dof': dof_results,
        'metrics_data': analysis_results
    }
    
    json_path = os.path.join(path, "analysis_data.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4, default=str)
    print(f"Raw data exported to: {json_path}")
    
    fig_final = create_summary_plot(z_indexes, lap_vals, ten_vals, wav_vals, dof_results)
    plt.show()
    plt.close('all')

    return True

# ENTRY POINT
def main():
    print("==================================================")
    print("   FOCUS ANALYSIS SYSTEM (POST-PROCESSING)        ")
    print("==================================================")
    
    folder_name = input("Enter the image directory name: ").strip()
    
    if not folder_name:
        print("Error: No valid directory name entered.")
        return

    start_time = time.time()
    success = analyze_stack(folder_name)
    end_time = time.time()
    
    print("==================================================")
    if success:
        print(f"Analysis completed successfully in {end_time - start_time:.2f} seconds.")
    else:
        print("Analysis finished with errors.")
    print("==================================================")

if __name__ == "__main__":
    main()