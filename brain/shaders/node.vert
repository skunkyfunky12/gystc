#version 330 core

layout(location = 0) in vec3 a_position;  // instanced, divisor=1
layout(location = 1) in vec3 a_color;     // instanced, divisor=1
layout(location = 2) in float a_size;     // instanced, divisor=1
layout(location = 3) in vec2 a_quad;      // per-vertex, quad corners [-1,-1] to [1,1]

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_time;

out vec3 v_color;
out vec2 v_uv;

void main()
{
    // Shader-based drift animation
    vec3 drift = vec3(
        sin(u_time * 0.5 + float(gl_InstanceID) * 0.7) * 0.8,
        cos(u_time * 0.3 + float(gl_InstanceID) * 1.1) * 0.6,
        sin(u_time * 0.4 + float(gl_InstanceID) * 0.9) * 0.5
    );

    // Extract billboard axes from view matrix
    vec3 cam_right = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 cam_up = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);

    // Billboard vertex position
    vec3 world_pos = a_position + drift + cam_right * a_quad.x * a_size + cam_up * a_quad.y * a_size;

    gl_Position = u_proj * u_view * vec4(world_pos, 1.0);

    v_color = a_color;
    v_uv = a_quad;
}
