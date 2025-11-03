// This Pine Script® code is subject to the Mozilla Public License 2.0: https://mozilla.org/MPL/2.0/
//@version=6
// ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
// ║                                              MARKOV CHAIN [3D]                                                                                                     ║
// ║                                                                                                                                                                    ║
// ║                                                                                                                                                                    ║
// ║  A lookback-based Markov Chain indicator that visualizes market regimes using transition probabilities                                                             ║
// ║  with animated 3D spheres representing Bull, Bear, and Neutral states and their transition probabilities                                                           ║  
// ║                                                                                                                                                                    ║
// ║  Features:                                                                                                                                                         ║
// ║  • Real-time regime detection using price returns or custom indicators                                                                                             ║
// ║  • Dynamic transition probability calculation                                                                                                                      ║
// ║  • Interactive 3D visualization with animated spheres                                                                                                              ║
// ║  • Bidirectional arrows showing state transitions                                                                                                                  ║
// ║  • Moving particle effects along transition paths                                                                                                                  ║
// ║                                                                                                                                                                    ║
// ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
// =============================================================================
// MARKOV CHAIN [3D] — Open‑Source
// Credits:
// - 3D sphere rendering technique inspired by @Alien_Algorithms
//   Source: https://www.tradingview.com/script/bT8hx615-Solar-System-in-3D-Astro-Tool-w-Zodiac/
// - Portions of Markov chain logic adapted from @RicardoSantos — MarkovChain (library)
//   Source: https://www.tradingview.com/script/gPZKYCjK-MarkovChain/
//
// Significant improvements:
// - Z‑score regime engine (price/custom features), hysteresis, bias output (+1/0/−1)
// - Rolling transition matrix with Laplace smoothing; updates only on confirmed bars
// - Advanced 3D animation (grid spheres, dynamic tilt/rotation) + particle flow arrows
// - input.source() integration for 3 normalized custom indicators; configurable visuals
indicator('Markov Chain [3D] | Fractalyst', overlay = false, max_polylines_count = 100, max_labels_count = 100, max_bars_back=500, scale = scale.right)
// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                      MODEL CONFIGURATION                                                        │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

model_group = "Markov Chain Model"
var int lookback_period = input.int(50, "Lookback Period", minval=5, maxval=100, group=model_group, tooltip="Lookback window for transition probabilities and Z-score baseline. Lower = more reactive; higher = smoother, more stable")
bool use_custom_indicators = input.bool(false, "Enable Custom Indicators", group=model_group, tooltip="Enable to use external indicators instead of price-based regime detection. When disabled, uses adaptive Z-score analysis of price returns")
ind1 = input.source(volume, "Custom Indicator 1", group=model_group, active = use_custom_indicators, tooltip="First custom indicator source for regime detection. Will be Z-score normalized over the lookback period")
ind2 = input.source(volume, "Custom Indicator 2", group=model_group, active = use_custom_indicators, tooltip="Second custom indicator source for regime detection. Will be Z-score normalized over the lookback period")
ind3 = input.source(volume, "Custom Indicator 3", group=model_group, active = use_custom_indicators, tooltip="Third custom indicator source for regime detection. Will be Z-score normalized over the lookback period")

// Market regime detection now uses Z-score statistical approach (see regime detection engine)

// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                      VISUAL CONFIGURATION                                                       │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

visual_group = 'Visual Settings'
var float scale_factor = input.float(800, 'Scale Factor', minval = 100, maxval = 1500, group = visual_group, tooltip="Controls overall size of the 3D visualization. Lower values for compact display, higher values for larger, more detailed spheres")
var int gui_shift = -int(input.int(200, 'Horizontal Shift', group = visual_group, tooltip="Radius of the 3D regime spheres. Affects sphere detail, arrow positioning, and particle animation paths"))
var float node_size = input.float(80, 'Node Size', minval = 30, maxval = 150, group = visual_group, tooltip="Horizontal position adjustment for the entire visualization. Useful for avoiding overlap with other indicators")

// Color settings
color_group = 'Color Settings'
var color bull_color = input.color(#00b9ff, 'Bullish', group = visual_group, tooltip="Color for the Bull market regime sphere and associated visual elements")
var color neutral_color = input.color(#787b86, 'Neutral', group = visual_group, tooltip="Color for the Neutral/Sideways market regime sphere and associated visual elements")
var color bear_color = input.color(#ff0051, 'Bearish', group = visual_group, tooltip="Color for the Bear market regime sphere and associated visual elements")
// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                   CUSTOM INDICATOR SETTINGS                                                     │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

// Normalized thresholds for custom indicators (±1 standard deviations)
var float ind1_bull_threshold = +1
var float ind1_bear_threshold = -1
var float ind2_bull_threshold = +1
var float ind2_bear_threshold = -1
var float ind3_bull_threshold = +1
var float ind3_bear_threshold = -1

// Equal weighting for indicator combination
var float ind1_weight = 1/3
var float ind2_weight = 1/3
var float ind3_weight = 1/3

// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                    INDICATOR NORMALIZATION                                                      │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

/// Normalizes indicator values using Z-score standardization
/// @param value Current indicator value
/// @param lookback_length Period for calculating mean and standard deviation
/// @returns Normalized value clamped between -3 and +3 standard deviations
normalize_indicator(float value, float lookback_length) =>
    // Calculate rolling statistics
    float mean_val = ta.sma(value, int(lookback_length))
    float stdev_val = ta.stdev(value, int(lookback_length))
    
    // Z-score normalization
    normalized = stdev_val > 0 ? (value - mean_val) / stdev_val : 0.0
    
    // Clamp to ±3 standard deviations to prevent extreme outliers
    math.max(-3.0, math.min(3.0, normalized))


/// Converts normalized indicator value to regime signal
/// @param normalized_value Z-score normalized indicator value
/// @param bull_threshold Threshold for bullish regime
/// @param bear_threshold Threshold for bearish regime
/// @returns 1.0 (Bull), -1.0 (Bear), or 0.0 (Neutral)
get_indicator_regime_signal(float normalized_value, float bull_threshold, float bear_threshold) =>
    if normalized_value > bull_threshold
        1.0  // Bullish regime
    else if normalized_value < bear_threshold
        -1.0 // Bearish regime
    else
        0.0  // Neutral regime


// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                     REGIME DETECTION ENGINE                                                     │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

/// Main regime detection function combining multiple indicators or price-based signals
/// @returns Market regime: 1 (Bull), -1 (Bear), 0 (Neutral)
get_combined_regime_signal() =>
    if use_custom_indicators
        // ── Custom Indicator Mode ──
        // Normalize all three indicators
        norm_ind1 = normalize_indicator(ind1, lookback_period)
        norm_ind2 = normalize_indicator(ind2, lookback_period)
        norm_ind3 = normalize_indicator(ind3, lookback_period)
        
        // Convert to regime signals
        signal1 = get_indicator_regime_signal(norm_ind1, ind1_bull_threshold, ind1_bear_threshold)
        signal2 = get_indicator_regime_signal(norm_ind2, ind2_bull_threshold, ind2_bear_threshold)
        signal3 = get_indicator_regime_signal(norm_ind3, ind3_bull_threshold, ind3_bear_threshold)
        
        // Weighted combination of signals
        total_weight = ind1_weight + ind2_weight + ind3_weight
        if total_weight > 0
            combined_signal = (signal1 * ind1_weight + signal2 * ind2_weight + signal3 * ind3_weight) / total_weight
            
            // Apply hysteresis to prevent noise
            if combined_signal > 0.05
                1   // Bullish consensus
            else if combined_signal < -0.05
                -1  // Bearish consensus
            else
                0   // Neutral/Mixed signals
        else
            0
    else
        // ── Price-Based Mode ──
        // Use Z-score of returns for statistically significant regime detection
        
        // Calculate percentage returns (more stable than absolute changes)
        returns = ta.change(close) / close[1]
        
        // Calculate rolling statistics for Z-score normalization
        lookback_len = lookback_period  // Use configured lookback period for statistical baseline
        returns_mean = ta.sma(returns, lookback_len)
        returns_stdev = ta.stdev(returns, lookback_len)
        
        // Calculate Z-score: (current_value - mean) / standard_deviation
        // Z-score tells us how many standard deviations away from normal
        z_score = returns_stdev > 0 ? (returns - returns_mean) / returns_stdev : 0
        
        // Statistical significance thresholds (balanced for practical regime detection)
        // Z-score > 0.33 = movement beyond ~63% of normal distribution
        // This gives roughly balanced regime distribution (~37% each for bull/bear, ~26% neutral)
        z_bull_threshold = +(1/3)   // Bullish when returns > 0.33 std dev above mean
        z_bear_threshold = -(1/3)   // Bearish when returns > 0.33 std dev below mean
        
        if z_score > z_bull_threshold
            1   // Statistically significant upward movement
        else if z_score < z_bear_threshold
            -1  // Statistically significant downward movement
        else
            0   // Normal range (within 1 standard deviation)

// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                      MARKET STATE TRACKING                                                     │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

// Get current market regime and prevent repainting on unconfirmed bars
market_state = get_combined_regime_signal()
market_state := barstate.isconfirmed ? market_state : market_state[1]


// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                    MARKOV CHAIN MATRICES                                                       │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

// Core Markov Chain data structures
// Matrix indices: 0=Bull, 1=Bear, 2=Neutral
var matrix<float> transition_matrix = matrix.new<float>(3, 3, 0.0)  // 3x3 transition count matrix
var array<int> state_counts = array.new<int>(3, 0)                 // State frequency counters
var int total_transitions = 0                                       // Total observed transitions
// Rolling window of recent confirmed states (as indices 0=Bull,1=Bear,2=Neutral)
var array<int> recent_states = array.new<int>()


// ── Transition Matrix Updates (bar-close, rolling window) ──
// Update counts only on confirmed bars and maintain a rolling window of size `lookback_period`
if barstate.isconfirmed and not na(market_state)
    // Map current market_state to index
    curr_state_idx = market_state == 1 ? 0 : market_state == -1 ? 1 : 2

    // Append current state; ensure window holds last `lookback_period`+1 states (for `lookback_period` transitions)
    array.push(recent_states, curr_state_idx)
    while array.size(recent_states) > lookback_period + 1
        array.shift(recent_states)

    // Rebuild matrices from the rolling window
    // Reset transition matrix to zeros
    for r = 0 to 2
        for c = 0 to 2
            matrix.set(transition_matrix, r, c, 0.0)
    // Reset state counts
    for s = 0 to 2
        array.set(state_counts, s, 0)

    // Recompute counts from recent_states
    sz = array.size(recent_states)
    if sz > 0
        // State occupancy over the window
        for i = 0 to sz - 1
            sidx = array.get(recent_states, i)
            array.set(state_counts, sidx, array.get(state_counts, sidx) + 1)
        // Transitions between consecutive states (sz-1 transitions)
        if sz > 1
            for i = 1 to sz - 1
                prev_idx = array.get(recent_states, i - 1)
                next_idx = array.get(recent_states, i)
                count_val = matrix.get(transition_matrix, prev_idx, next_idx)
                matrix.set(transition_matrix, prev_idx, next_idx, count_val + 1)
        total_transitions := sz > 0 ? sz - 1 : 0


/// Calculates transition probability from one state to another
/// @param from_state Source state index (0=Bull, 1=Bear, 2=Neutral)
/// @param to_state Target state index (0=Bull, 1=Bear, 2=Neutral)
/// @returns Probability of transitioning from source to target state
get_transition_probability(int from_state, int to_state) =>
    // Calculate row sum (total transitions from source state)
    // Apply Laplace smoothing (+1 per outcome) for stability with sparse data
    row_sum = 0.0
    for j = 0 to 2
        row_sum += (matrix.get(transition_matrix, from_state, j) + 1.0)

    // With Laplace smoothing, row_sum cannot be zero; still keep guard for safety
    if row_sum == 0
        0.0
    else
        (matrix.get(transition_matrix, from_state, to_state) + 1.0) / row_sum


// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                      3D GRAPHICS ENGINE                                                       │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

/// 3D point structure for sphere generation and transformations
type Point3D
	float x  // X coordinate in 3D space
	float y  // Y coordinate in 3D space
	float z  // Z coordinate in 3D space

// ── 3D Rendering Constants ──
var float DEG_TO_RAD = math.pi / 180          // Degree to radian conversion
var float CAMERA_ANGLE = 45.0                 // Camera viewing angle
var float CAMERA_COS = math.cos(CAMERA_ANGLE * DEG_TO_RAD)  // Precomputed cosine
var float CAMERA_SIN = math.sin(CAMERA_ANGLE * DEG_TO_RAD)  // Precomputed sine
var int SEGMENTS = 12                         // Sphere tessellation resolution


/// Generates 3D sphere vertices using spherical coordinates
/// @param radius Sphere radius
/// @returns Array of 3D points forming a sphere
f_generate_sphere_points(radius) =>
    points = array.new<Point3D>()
    // Generate sphere using latitude-longitude parameterization
    for i = 0 to SEGMENTS by 1
        lat = math.pi * (-0.5 + i / SEGMENTS)  // Latitude: -π/2 to π/2
        for j = 0 to SEGMENTS by 1
            lon = 2 * math.pi * j / SEGMENTS   // Longitude: 0 to 2π
            // Convert spherical to Cartesian coordinates
            x = radius * math.cos(lat) * math.cos(lon)
            y = radius * math.sin(lat)
            z = radius * math.cos(lat) * math.sin(lon)
            array.push(points, Point3D.new(x, y, z))
    points


/// Applies X-axis rotation to a 3D point (camera perspective)
/// @param point Input 3D point
/// @returns Rotated 3D point
rotate_x(Point3D point) =>
    // Rotation matrix around X-axis
    y = point.y * CAMERA_COS - point.z * CAMERA_SIN
    z = point.y * CAMERA_SIN + point.z * CAMERA_COS
    Point3D.new(point.x, y, z)


/// Transforms and projects 3D sphere points to 2D chart coordinates
/// @param points Array of 3D sphere vertices
/// @param center_x X position of sphere center
/// @param center_y Y position of sphere center  
/// @param center_z Z position of sphere center
/// @param tilt Tilt angle in degrees
/// @param rotation Rotation angle in radians
/// @returns Array of 2D chart points ready for rendering
update_sphere(array<Point3D> points, float center_x, float center_y, float center_z, float tilt, float rotation) =>
    projected_points = array.new<chart.point>()
    
    // Precompute trigonometric values for performance
    cos_rotation = math.cos(rotation)
    sin_rotation = math.sin(rotation)
    cos_tilt = math.cos(tilt * DEG_TO_RAD)
    sin_tilt = math.sin(tilt * DEG_TO_RAD)

    for point in points
        // Apply Y-axis rotation (spinning animation)
        rotated_x = point.x * cos_rotation - point.z * sin_rotation
        rotated_z = point.x * sin_rotation + point.z * cos_rotation

        // Apply X-axis tilt (3D perspective effect)
        tilted_y = point.y * cos_tilt - rotated_z * sin_tilt
        tilted_z_final = point.y * sin_tilt + rotated_z * cos_tilt

        // Project to 2D screen coordinates
        projected_x = rotated_x + center_x
        projected_y = tilted_y + center_y

        array.push(projected_points, chart.point.from_index(gui_shift + bar_index + int(projected_x), projected_y + scale_factor))
    projected_points


/// Renders 3D sphere using polylines for latitude and longitude grid
/// @param projected_points Array of 2D chart points from sphere projection
/// @param col Color for sphere wireframe
draw_3d_sphere(array<chart.point> projected_points, color col, int poly_width = 1) =>
    // Draw latitude lines (horizontal circles)
    for i = 0 to SEGMENTS by 1
        lat_points = array.new<chart.point>()
        for j = 0 to SEGMENTS by 1
            idx = i * (SEGMENTS + 1) + j
            if idx < array.size(projected_points)
                array.push(lat_points, array.get(projected_points, idx))
        if array.size(lat_points) > 1
            polyline.new(lat_points, line_color = col, line_width = poly_width)
    
    // Draw longitude lines (vertical curves)
    for j = 0 to SEGMENTS by 1
        lon_points = array.new<chart.point>()
        for i = 0 to SEGMENTS by 1
            idx = i * (SEGMENTS + 1) + j
            if idx < array.size(projected_points)
                array.push(lon_points, array.get(projected_points, idx))
        if array.size(lon_points) > 1
            polyline.new(lon_points, line_color = col, line_width = poly_width)


// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                     PARTICLE ANIMATION SYSTEM                                                  │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

/// Creates animated particle flow between two points with color gradients
/// @param start_x Starting X coordinate
/// @param start_y Starting Y coordinate
/// @param end_x Ending X coordinate
/// @param end_y Ending Y coordinate
/// @param animation_time Current animation time for phase calculation
/// @param start_color Color at animation start
/// @param end_color Color at animation end
/// @param offset_x Additional X offset for positioning
/// @param offset_y Additional Y offset for positioning
draw_moving_circles(float start_x, float start_y, float end_x, float end_y, float animation_time, color start_color, color end_color, float offset_x, float offset_y) =>
    // Number of particles in the animation stream
    num_circles = 6
    
    // Calculate direction vector and length
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx * dx + dy * dy)
    
    // Calculate perpendicular vector for alternating particle paths
    perp_x = length > 0 ? -dy / length : 0
    perp_y = length > 0 ? dx / length : 0
    
    // Offset distance for alternating sides
    edge_offset = 1.5
    
    // Generate animated particles with staggered phases
    for i = 0 to num_circles - 1
        // Create unique phase offset for each particle
        phase_offset = i * (2 * math.pi / num_circles)
        
        // Calculate sinusoidal oscillation for smooth animation
        oscillation = math.sin(animation_time * 2 + phase_offset)
        
        // Convert oscillation (-1 to 1) to progress (0 to 1)
        progress = (oscillation + 1) / 2
        
        // Calculate particle position along the path
        base_x = start_x + (end_x - start_x) * progress + offset_x
        base_y = start_y + (end_y - start_y) * progress + offset_y
        
        // Alternate particles between two parallel paths
        side_multiplier = (i % 2 == 0) ? 1 : -1
        circle_x = base_x + perp_x * edge_offset * side_multiplier
        circle_y = base_y + perp_y * edge_offset * side_multiplier
        
        // Extract RGB components for gradient calculation
        start_r = color.r(start_color)
        start_g = color.g(start_color)
        start_b = color.b(start_color)
        
        end_r = color.r(end_color)
        end_g = color.g(end_color)
        end_b = color.b(end_color)
        
        // Interpolate colors based on particle progress
        gradient_r = start_r + (end_r - start_r) * progress
        gradient_g = start_g + (end_g - start_g) * progress
        gradient_b = start_b + (end_b - start_b) * progress
        
        // Create final gradient color
        gradient_color = color.rgb(gradient_r, gradient_g, gradient_b)
        
        // Render particle as colored circle
        label.new(gui_shift + bar_index + int(circle_x), circle_y + scale_factor, 
                  "", 
                  color=gradient_color, 
                  size= 1, 
                  style=label.style_circle, 
                  xloc=xloc.bar_index)


// ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                    MARKET REGIME NODE RENDERING                                                │
// └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

/// Renders an animated 3D sphere representing a market regime state
/// @param x X position of the node center
/// @param y Y position of the node center
/// @param label_text Text label for the node
/// @param node_color Primary color for the node
/// @param prob Probability value to display
/// @param rotation Animation rotation time
/// @param node_type Type of node ("Bull", "Bear", "Neutral")
/// @param current_market_state Current active market state
draw_market_node(float x, float y, string label_text, color node_color, float prob, float rotation, string node_type, int current_market_state) =>
    // Generate base sphere geometry
    sphere_points = f_generate_sphere_points(node_size * 0.8)

    // Calculate complex rotation animations for organic movement
    base_rotation = rotation * 0.8
    x_rotation = rotation * 0.6 + (x * 0.01)
    y_rotation = rotation * 1.2 + (y * 0.01)
    z_rotation = rotation * 0.9 + math.sin(rotation * 0.3) * 0.5
    
    // Dynamic tilt with multiple harmonic components for natural movement
    tilt = 15.0 + 8.0 * math.sin(rotation * 0.4) + 5.0 * math.cos(rotation * 0.7) + 3.0 * math.sin(rotation * 1.1)
    
    // Combine rotation components for complex 3D animation
    combined_rotation = base_rotation + math.sin(x_rotation) * 0.3 + math.cos(y_rotation) * 0.2

    // Project 3D sphere to 2D chart coordinates
    projected_points = update_sphere(sphere_points, x, y, 0, tilt, combined_rotation)


    // ── Active State Detection ──
    // Determine if this node represents the currently active market regime
    should_have_inner_circle = false
    market_sphere_index = market_state == 1 ? 0 : (market_state == -1 ? 1 : 2)
    sphere_index = node_type == 'Bull' ? 0 : (node_type == 'Bear' ? 1 : 2)
    should_have_inner_circle := (market_sphere_index == sphere_index)
    
    // ── Dynamic Color Calculation ──
    color sphere_color = na
    if should_have_inner_circle
        // Active state: High opacity, bright appearance
        sphere_color := color.new(node_color, 20)
    else
        // Inactive states: Apply special effects based on node type
        if node_type == 'Neutral'
            // Neutral node gets animated lighting effect
            light_intensity = math.sin(rotation * 1.5) * 0.3 + 0.7
            
            // Extract base gray color components
            base_gray_r = color.r(node_color)
            base_gray_g = color.g(node_color)
            base_gray_b = color.b(node_color)
            
            // Calculate color influences for warm lighting
            blue_influence = 15 * light_intensity
            red_influence = 12 * light_intensity
            
            // Apply lighting to create dynamic color shifting
            lit_r = math.min(255, base_gray_r + red_influence)
            lit_g = math.min(255, base_gray_g + (blue_influence + red_influence) * 0.3)
            lit_b = math.min(255, base_gray_b + blue_influence)
            
            // Create final lit color with transparency
            lit_gray_color = color.rgb(lit_r, lit_g, lit_b)
            sphere_color := color.new(lit_gray_color, 60)
        else
            // Bull/Bear nodes: Standard dimmed appearance when inactive
            sphere_color := color.new(node_color, 60)
    // ── Render Main Sphere ──
    draw_3d_sphere(projected_points, sphere_color)
    
    // ── Active State Inner Sphere Animation ──
    if should_have_inner_circle
        // Create pulsing animation for active state indicator
        pulse_factor = 0.15 * math.sin(rotation * 3.0)
        
        // Dynamic transparency based on pulse
        circle_transparency = int(5 + math.abs(pulse_factor * 10))
        
        // Create pulsing inner sphere color
        inner_circle_color = color.new(node_color, circle_transparency)
        
        // Calculate pulsing size (30% base + pulse variation)
        inner_circle_size = node_size * (0.3 + pulse_factor)
        inner_circle_points = f_generate_sphere_points(inner_circle_size)
        inner_circle_projected = update_sphere(inner_circle_points, x, y, 0, tilt, combined_rotation * 1.5)
        draw_3d_sphere(inner_circle_projected, inner_circle_color, 2)
    
    // ── Render Node Labels ──

    if node_type == 'Bull' or node_type == 'Bear'

        label_y = y + node_size * 0.85
        
        // Determine label transparency based on inner circle presence (same logic as sphere)
        label_text_transparency = should_have_inner_circle ? 0 : 30  // Bright when active, dimmed when inactive
        

        label.new(gui_shift + bar_index + int(x), label_y + scale_factor, 
                  str.tostring(prob * 100, '0.00') + '%', 
                  color=color.new(color.white, 100), textcolor=color.new(node_color, label_text_transparency), 
                  size=size.normal, style=label.style_label_down, xloc=xloc.bar_index, 
                  text_formatting=text.format_bold)
    else if node_type == 'Neutral'

        label_y = y - node_size * 0.85
        
        // Determine label transparency based on inner circle presence (same logic as sphere)
        label_text_transparency = should_have_inner_circle ? 0 : 30  // Bright when active, dimmed when inactive
        

        label.new(gui_shift + bar_index + int(x), label_y + scale_factor, 
                  str.tostring(prob * 100, '0.00') + '%', 
                  color=color.new(color.white, 100), textcolor=color.new(node_color, label_text_transparency), 
                  size=size.normal, style=label.style_label_up, xloc=xloc.bar_index, 
                  text_formatting=text.format_bold)


// ═══════════════════════════════════════════════════════════════════════════
// ███ BIDIRECTIONAL TRANSITION ARROW VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════════
// Renders animated bidirectional arrows between market regime nodes to visualize
// transition probabilities in both directions. Arrows are offset to prevent overlap
// and include animated segments with probability labels.
//
// Parameters:
// • start_x/y, end_x/y: Coordinates of connected regime nodes
// • prob1, prob2: Transition probabilities for each direction
// • start_color, end_color: Colors of the connected regimes
// • connection_type: Layout type (horizontal, diagonal) for label positioning
// • animation_time: Time factor for animated effects
// ═══════════════════════════════════════════════════════════════════════════
draw_bidirectional_arrows(start_x, start_y, end_x, end_y, prob1, prob2, start_color, end_color, connection_type, animation_time) =>

    // ── Vector Calculations ──
    // Calculate direction vector between nodes
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx * dx + dy * dy)
    
    // Normalize direction vector for unit length
    norm_dx = dx / length
    norm_dy = dy / length
    
    // Calculate perpendicular vector for arrow offset
    // Perpendicular to (dx, dy) is (-dy, dx)
    perp_x = -norm_dy
    perp_y = norm_dx
    
    // ── Arrow Offset Configuration ──
    // Offset arrows to prevent overlap (20% of node size)
    offset = node_size * 0.2
    
    // ── Sphere Boundary Adjustment ──
    // Adjust arrow endpoints to start/end at sphere boundaries
    // rather than at center points (90% of node radius)
    sphere_radius = node_size * 0.9
    adj_start_x = start_x + norm_dx * sphere_radius
    adj_start_y = start_y + norm_dy * sphere_radius
    adj_end_x = end_x - norm_dx * sphere_radius
    adj_end_y = end_y - norm_dy * sphere_radius
    
    // ── First Arrow Path Calculation ──
    // Offset in positive perpendicular direction
    offset_start_x1 = adj_start_x + perp_x * offset
    offset_start_y1 = adj_start_y + perp_y * offset
    offset_end_x1 = adj_end_x + perp_x * offset
    offset_end_y1 = adj_end_y + perp_y * offset
    
    // ── Second Arrow Path Calculation ──
    // Offset in negative perpendicular direction (opposite side)
    offset_start_x2 = adj_end_x - perp_x * offset
    offset_start_y2 = adj_end_y - perp_y * offset
    offset_end_x2 = adj_start_x - perp_x * offset
    offset_end_y2 = adj_start_y - perp_y * offset
    

    // ── Arrow Segmentation Configuration ──
    // Divide arrows into segments for animated/styled rendering
    num_segments = 8
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ▶ FIRST DIRECTIONAL ARROW RENDERING
    // ═══════════════════════════════════════════════════════════════════════════
    for i = 0 to num_segments - 1
        // Calculate segment position (0 to 1 normalized)
        segment_start = i / num_segments
        segment_end = (i + 1) / num_segments
        
        // ── Segment Coordinate Interpolation ──
        // Linear interpolation along arrow path
        seg_start_x = offset_start_x1 + (offset_end_x1 - offset_start_x1) * segment_start
        seg_start_y = offset_start_y1 + (offset_end_y1 - offset_start_y1) * segment_start
        seg_end_x = offset_start_x1 + (offset_end_x1 - offset_start_x1) * segment_end
        seg_end_y = offset_start_y1 + (offset_end_y1 - offset_start_y1) * segment_end
        
        // ── Segment Styling ──
        // Semi-transparent foreground color for visibility
        segment_color = color.new(chart.fg_color, 35)
        
        // Add arrowhead to final segment
        line_style = (i == num_segments - 1) ? line.style_arrow_right : line.style_solid
        
        // Draw segment line
        line.new(gui_shift + bar_index + int(seg_start_x), seg_start_y + scale_factor,
                 gui_shift + bar_index + int(seg_end_x), seg_end_y + scale_factor,
                 color=segment_color, style=line_style, width=2, xloc=xloc.bar_index)
    

    // ═══════════════════════════════════════════════════════════════════════════
    // ◀ SECOND DIRECTIONAL ARROW RENDERING (REVERSE)
    // ═══════════════════════════════════════════════════════════════════════════
    for i = 0 to num_segments - 1
        // Calculate segment position (0 to 1 normalized)
        segment_start = i / num_segments
        segment_end = (i + 1) / num_segments
        
        // ── Segment Coordinate Interpolation ──
        // Linear interpolation along reverse arrow path
        seg_start_x = offset_start_x2 + (offset_end_x2 - offset_start_x2) * segment_start
        seg_start_y = offset_start_y2 + (offset_end_y2 - offset_start_y2) * segment_start
        seg_end_x = offset_start_x2 + (offset_end_x2 - offset_start_x2) * segment_end
        seg_end_y = offset_start_y2 + (offset_end_y2 - offset_start_y2) * segment_end
        
        // ── Segment Styling ──
        // Matching style with first arrow for consistency
        segment_color = color.new(chart.fg_color, 35)
        
        // Add arrowhead to final segment
        line_style = (i == num_segments - 1) ? line.style_arrow_right : line.style_solid
        
        // Draw segment line
        line.new(gui_shift + bar_index + int(seg_start_x), seg_start_y + scale_factor,
                 gui_shift + bar_index + int(seg_end_x), seg_end_y + scale_factor,
                 color=segment_color, style=line_style, width=2, xloc=xloc.bar_index)
    

    // ═══════════════════════════════════════════════════════════════════════════
    // 🏷️ TRANSITION PROBABILITY LABELS
    // ═══════════════════════════════════════════════════════════════════════════
    
    // ── Label Position Calculation ──
    // Place labels at midpoint of each arrow
    mid_x1 = (offset_start_x1 + offset_end_x1) / 2
    mid_y1 = (offset_start_y1 + offset_end_y1) / 2
    mid_x2 = (offset_start_x2 + offset_end_x2) / 2
    mid_y2 = (offset_start_y2 + offset_end_y2) / 2
    
    // ── Dynamic Label Style Selection ──
    // Adjust label orientation based on connection type
    first_label_style = label.style_label_up
    second_label_style = label.style_label_down
    
    if connection_type == "horizontal"
        // For horizontal connections, place labels above/below arrows
        upper_arrow_y = math.max(mid_y1, mid_y2)
        first_label_style := mid_y1 == upper_arrow_y ? label.style_label_down : label.style_label_up
        second_label_style := mid_y2 == upper_arrow_y ? label.style_label_down : label.style_label_up
    else if connection_type == "left_diagonal"
        // For left diagonal, use corner label styles
        first_label_style := label.style_label_lower_left
        second_label_style := label.style_label_upper_right
    else if connection_type == "right_diagonal"
        // For right diagonal, use opposite corner styles
        first_label_style := label.style_label_upper_left
        second_label_style := label.style_label_lower_right
    
    // ── First Arrow Probability Label ──
    // Display transition probability as percentage
    label.new(gui_shift + bar_index + int(mid_x1), mid_y1 + scale_factor, 
              str.tostring(prob1 * 100, '0.00') + '%', 
              color=color.new(color.white, 100), textcolor= color.new(chart.fg_color,5), size=size.normal, 
              style=first_label_style, xloc=xloc.bar_index, text_formatting = text.format_bold)
    
    // ── Second Arrow Probability Label ──
    // Display reverse transition probability
    label.new(gui_shift + bar_index + int(mid_x2), mid_y2 + scale_factor,
              str.tostring(prob2 * 100, '0.00') + '%', 
              color=color.new(color.white, 100), textcolor= color.new(chart.fg_color,5), size=size.normal, 
              style=second_label_style, xloc=xloc.bar_index, text_formatting = text.format_bold)




// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ 🎨 MAIN MARKOV CHAIN VISUALIZATION ENGINE                                  ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// Master function that orchestrates the entire 3D Markov Chain visualization.
// Combines all visual elements: animated spheres, transition arrows, particles,
// and probability labels to create an interactive market regime display.
//
// Parameters:
// • rotation: Global animation time factor for synchronized animations
// ═══════════════════════════════════════════════════════════════════════════
draw_markov_chain_with_regime_data(float rotation) =>

    // ── Clear Previous Frame ──
    // Remove all visual elements from previous render
    for poly in polyline.all
        polyline.delete(poly)
    for lbl in label.all
        label.delete(lbl)
    for ln in line.all
        line.delete(ln)

    // Show warming-up notice until we have at least one transition (2 states)
    warmup_sz = array.size(recent_states)
    if warmup_sz < 2
        label.new(gui_shift + bar_index, scale_factor + 20,
                  "Warming up: collecting transitions...",
                  color = color.new(color.white, 100), textcolor = chart.fg_color,
                  xloc = xloc.bar_index, style = label.style_none, size = size.small,
                  text_formatting = text.format_bold)

    // ═══════════════════════════════════════════════════════════════════════════
    // 📍 NODE POSITION CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════
    // Fixed positions for regime nodes in triangular arrangement
    bull_x = -150.0    // Left position
    bull_y = 50.0      // Upper level
    bear_x = 150.0     // Right position
    bear_y = 50.0      // Upper level
    neutral_x = 0.0    // Center position
    neutral_y = -100.0 // Lower level

    // ═══════════════════════════════════════════════════════════════════════════
    // 📊 STATE PROBABILITY CALCULATIONS
    // ═══════════════════════════════════════════════════════════════════════════
    // Calculate empirical probabilities from historical state counts
    total_states = array.get(state_counts, 0) + array.get(state_counts, 1) + array.get(state_counts, 2)
    bull_prob = total_states > 0 ? array.get(state_counts, 0) / total_states : 1.0/3.0
    bear_prob = total_states > 0 ? array.get(state_counts, 1) / total_states : 1.0/3.0
    neutral_prob = total_states > 0 ? array.get(state_counts, 2) / total_states : 1.0/3.0

    // ═══════════════════════════════════════════════════════════════════════════
    // 🔄 TRANSITION PROBABILITY EXTRACTION
    // ═══════════════════════════════════════════════════════════════════════════
    // Extract all pairwise transition probabilities from Markov matrix
    bull_to_bear_prob = get_transition_probability(0, 1)     // Bull → Bear
    bear_to_bull_prob = get_transition_probability(1, 0)     // Bear → Bull
    bull_to_neutral_prob = get_transition_probability(0, 2)  // Bull → Neutral
    neutral_to_bull_prob = get_transition_probability(2, 0)  // Neutral → Bull
    bear_to_neutral_prob = get_transition_probability(1, 2)  // Bear → Neutral
    neutral_to_bear_prob = get_transition_probability(2, 1)  // Neutral → Bear
    bull_self_prob = get_transition_probability(0, 0)       // Bull → Bull (self)
    bear_self_prob = get_transition_probability(1, 1)       // Bear → Bear (self)
    neutral_self_prob = get_transition_probability(2, 2)     // Neutral → Neutral (self)

    // ═══════════════════════════════════════════════════════════════════════════
    // 🔮 3D MARKET REGIME NODE RENDERING
    // ═══════════════════════════════════════════════════════════════════════════
    // Render animated 3D spheres for each market regime
    // Active regime has pulsing inner sphere and brighter colors
    draw_market_node(bull_x, bull_y, 'Bull\nmarket', bull_color, bull_self_prob, rotation, 'Bull', market_state)
    draw_market_node(bear_x, bear_y, 'Bear\nmarket', bear_color, bear_self_prob, rotation, 'Bear', market_state)
    draw_market_node(neutral_x, neutral_y, 'Neutral\nmarket', neutral_color, neutral_self_prob, rotation, 'Neutral', market_state)

    // ═══════════════════════════════════════════════════════════════════════════
    // ↔️ TRANSITION ARROW RENDERING
    // ═══════════════════════════════════════════════════════════════════════════
    // Draw bidirectional arrows between all regime pairs
    // Bull ↔ Bear (horizontal layout)
    draw_bidirectional_arrows(bull_x, bull_y, bear_x, bear_y, bull_to_bear_prob, bear_to_bull_prob, bull_color, bear_color, "horizontal", rotation)
    
    // Bull ↔ Neutral (left diagonal layout)
    draw_bidirectional_arrows(bull_x, bull_y, neutral_x, neutral_y, bull_to_neutral_prob, neutral_to_bull_prob, bull_color, neutral_color, "left_diagonal", rotation)
    
    // Bear ↔ Neutral (right diagonal layout)
    draw_bidirectional_arrows(bear_x, bear_y, neutral_x, neutral_y, bear_to_neutral_prob, neutral_to_bear_prob, bear_color, neutral_color, "right_diagonal", rotation)
    

    // ═══════════════════════════════════════════════════════════════════════════
    // ✨ PARTICLE ANIMATION SYSTEM
    // ═══════════════════════════════════════════════════════════════════════════
    // Animated particles flow along transition paths to visualize
    // the dynamic nature of regime transitions. Particles use color
    // gradients from source to destination regime.
    
    // ── Particle Path Configuration ──
    arrow_color = color.new(chart.fg_color,35)
    offset = node_size * 0.2        // Path offset to align with arrows
    sphere_radius = node_size * 0.9  // Start/end at sphere boundaries
    
    // ════════════════════════════════════════════════════════════════════
    // 🔵↔️🔴 BULL-BEAR PARTICLE ANIMATIONS
    // ════════════════════════════════════════════════════════════════════
    // Calculate normalized direction vectors for Bull-Bear connection
    bull_bear_dx = bear_x - bull_x
    bull_bear_dy = bear_y - bull_y
    bull_bear_length = math.sqrt(bull_bear_dx * bull_bear_dx + bull_bear_dy * bull_bear_dy)
    bull_bear_norm_dx = bull_bear_dx / bull_bear_length
    bull_bear_norm_dy = bull_bear_dy / bull_bear_length
    bull_bear_perp_x = -bull_bear_norm_dy  // Perpendicular for offset
    bull_bear_perp_y = bull_bear_norm_dx
    
    // ── Forward Path (Bull → Bear) ──
    // Calculate particle path with positive offset
    bull_bear_start_x1 = bull_x + bull_bear_norm_dx * sphere_radius + bull_bear_perp_x * offset
    bull_bear_start_y1 = bull_y + bull_bear_norm_dy * sphere_radius + bull_bear_perp_y * offset
    bull_bear_end_x1 = bear_x - bull_bear_norm_dx * sphere_radius + bull_bear_perp_x * offset
    bull_bear_end_y1 = bear_y - bull_bear_norm_dy * sphere_radius + bull_bear_perp_y * offset
    
    // ── Reverse Path (Bear → Bull) ──
    // Calculate particle path with negative offset
    bull_bear_start_x2 = bear_x - bull_bear_norm_dx * sphere_radius - bull_bear_perp_x * offset
    bull_bear_start_y2 = bear_y - bull_bear_norm_dy * sphere_radius - bull_bear_perp_y * offset
    bull_bear_end_x2 = bull_x + bull_bear_norm_dx * sphere_radius - bull_bear_perp_x * offset
    bull_bear_end_y2 = bull_y + bull_bear_norm_dy * sphere_radius - bull_bear_perp_y * offset
    
    // Render animated particles with color gradients
    draw_moving_circles(bull_bear_start_x1, bull_bear_start_y1, bull_bear_end_x1, bull_bear_end_y1, rotation, bull_color, bear_color, 0, 0)
    draw_moving_circles(bull_bear_start_x2, bull_bear_start_y2, bull_bear_end_x2, bull_bear_end_y2, rotation, bear_color, bull_color, 0, 0)
    

    // ════════════════════════════════════════════════════════════════════
    // 🔵↔️⚪ BULL-NEUTRAL PARTICLE ANIMATIONS
    // ════════════════════════════════════════════════════════════════════
    // Calculate normalized direction vectors for Bull-Neutral connection
    bull_neutral_dx = neutral_x - bull_x
    bull_neutral_dy = neutral_y - bull_y
    bull_neutral_length = math.sqrt(bull_neutral_dx * bull_neutral_dx + bull_neutral_dy * bull_neutral_dy)
    bull_neutral_norm_dx = bull_neutral_dx / bull_neutral_length
    bull_neutral_norm_dy = bull_neutral_dy / bull_neutral_length
    bull_neutral_perp_x = -bull_neutral_norm_dy  // Perpendicular for offset
    bull_neutral_perp_y = bull_neutral_norm_dx
    
    // ── Forward Path (Bull → Neutral) ──
    // Calculate particle path with positive offset
    bull_neutral_start_x1 = bull_x + bull_neutral_norm_dx * sphere_radius + bull_neutral_perp_x * offset
    bull_neutral_start_y1 = bull_y + bull_neutral_norm_dy * sphere_radius + bull_neutral_perp_y * offset
    bull_neutral_end_x1 = neutral_x - bull_neutral_norm_dx * sphere_radius + bull_neutral_perp_x * offset
    bull_neutral_end_y1 = neutral_y - bull_neutral_norm_dy * sphere_radius + bull_neutral_perp_y * offset
    
    // ── Reverse Path (Neutral → Bull) ──
    // Calculate particle path with negative offset
    bull_neutral_start_x2 = neutral_x - bull_neutral_norm_dx * sphere_radius - bull_neutral_perp_x * offset
    bull_neutral_start_y2 = neutral_y - bull_neutral_norm_dy * sphere_radius - bull_neutral_perp_y * offset
    bull_neutral_end_x2 = bull_x + bull_neutral_norm_dx * sphere_radius - bull_neutral_perp_x * offset
    bull_neutral_end_y2 = bull_y + bull_neutral_norm_dy * sphere_radius - bull_neutral_perp_y * offset
    
    // Render animated particles with color gradients
    draw_moving_circles(bull_neutral_start_x1, bull_neutral_start_y1, bull_neutral_end_x1, bull_neutral_end_y1, rotation, bull_color, neutral_color, 0, 0)
    draw_moving_circles(bull_neutral_start_x2, bull_neutral_start_y2, bull_neutral_end_x2, bull_neutral_end_y2, rotation, neutral_color, bull_color, 0, 0)
    

    // ════════════════════════════════════════════════════════════════════
    // 🔴↔️⚪ BEAR-NEUTRAL PARTICLE ANIMATIONS
    // ════════════════════════════════════════════════════════════════════
    // Calculate normalized direction vectors for Bear-Neutral connection
    bear_neutral_dx = neutral_x - bear_x
    bear_neutral_dy = neutral_y - bear_y
    bear_neutral_length = math.sqrt(bear_neutral_dx * bear_neutral_dx + bear_neutral_dy * bear_neutral_dy)
    bear_neutral_norm_dx = bear_neutral_dx / bear_neutral_length
    bear_neutral_norm_dy = bear_neutral_dy / bear_neutral_length
    bear_neutral_perp_x = -bear_neutral_norm_dy  // Perpendicular for offset
    bear_neutral_perp_y = bear_neutral_norm_dx
    
    // ── Forward Path (Bear → Neutral) ──
    // Calculate particle path with positive offset
    bear_neutral_start_x1 = bear_x + bear_neutral_norm_dx * sphere_radius + bear_neutral_perp_x * offset
    bear_neutral_start_y1 = bear_y + bear_neutral_norm_dy * sphere_radius + bear_neutral_perp_y * offset
    bear_neutral_end_x1 = neutral_x - bear_neutral_norm_dx * sphere_radius + bear_neutral_perp_x * offset
    bear_neutral_end_y1 = neutral_y - bear_neutral_norm_dy * sphere_radius + bear_neutral_perp_y * offset
    
    // ── Reverse Path (Neutral → Bear) ──
    // Calculate particle path with negative offset
    bear_neutral_start_x2 = neutral_x - bear_neutral_norm_dx * sphere_radius - bear_neutral_perp_x * offset
    bear_neutral_start_y2 = neutral_y - bear_neutral_norm_dy * sphere_radius - bear_neutral_perp_y * offset
    bear_neutral_end_x2 = bear_x + bear_neutral_norm_dx * sphere_radius - bear_neutral_perp_x * offset
    bear_neutral_end_y2 = bear_y + bear_neutral_norm_dy * sphere_radius - bear_neutral_perp_y * offset
    
    // Render animated particles with color gradients
    draw_moving_circles(bear_neutral_start_x1, bear_neutral_start_y1, bear_neutral_end_x1, bear_neutral_end_y1, rotation, bear_color, neutral_color, 0, 0)
    draw_moving_circles(bear_neutral_start_x2, bear_neutral_start_y2, bear_neutral_end_x2, bear_neutral_end_y2, rotation, neutral_color, bear_color, 0, 0)


// ═══════════════════════════════════════════════════════════════════════════
// ⚙️ ANIMATION ENGINE & MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════════════════
// Controls the animation timing and triggers the main visualization rendering
// on the last bar. The rotation_time variable provides synchronized animation
// across all visual elements.

// ── Animation State Variables ──
var bool init = false              // Initialization flag (unused in current version)
varip float rotation_time = 0.0    // Global animation timer (persistent across bars)

// ── Main Rendering Loop ──
// Execute visualization only on the last bar to optimize performance
if barstate.islast
    // Increment animation timer for smooth continuous rotation
    rotation_time := rotation_time + 0.05
    
    // Render complete Markov Chain visualization
    draw_markov_chain_with_regime_data(rotation_time)

// ═══════════════════════════════════════════════════════════════════════════
// 📊 DATA WINDOW OUTPUT
// ═══════════════════════════════════════════════════════════════════════════
// Export current market state to data window for external analysis
color regime_color = market_state == 1 ? bull_color : market_state == -1 ? bear_color : neutral_color
plot(market_state, "Current Market State", regime_color, linewidth=2,display = display.data_window, editable = false)
// Plot market regime on candles
plotcandle(open,high,low,close,"Regime state",regime_color,regime_color, force_overlay = true,editable = true,display = display.none)