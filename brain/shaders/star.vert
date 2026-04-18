#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in float a_brightness;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_time;

out float v_brightness;

void main()
{
    gl_Position = u_proj * u_view * vec4(a_position, 1.0);

    // Twinkling: each star has its own phase
    float phase = dot(a_position, vec3(0.17, 0.31, 0.53));
    float twinkle = 0.7 + 0.3 * sin(u_time * 2.0 + phase * 100.0);

    gl_PointSize = 1.2 + a_brightness * 1.8 * twinkle;
    v_brightness = a_brightness * twinkle;
}
