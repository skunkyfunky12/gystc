#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in float a_brightness;

uniform mat4 u_view;
uniform mat4 u_proj;

out float v_brightness;

void main()
{
    gl_Position = u_proj * u_view * vec4(a_position, 1.0);
    gl_PointSize = 1.5;
    v_brightness = a_brightness;
}
