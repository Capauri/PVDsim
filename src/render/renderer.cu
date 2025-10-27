#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "renderer.hpp"
#include "particle.hpp"

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

static GLuint cubeVAO = 0, cubeVBO = 0, cubeEBO = 0;
static GLuint ptsVAO = 0, ptsVBO = 0;
static GLuint lineProg = 0, pointProg = 0;

static GLuint compileShader(const char* vsSrc, const char* fsSrc) {
    auto compile = [&](GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint ok;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[512];
            glGetShaderInfoLog(shader, 512, nullptr, buf);
            std::cerr << "Shader compile error:\n" << buf << std::endl;
            glDeleteShader(shader);
            return 0u;
        }
        return shader;
        };

    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char buf[512];
        glGetProgramInfoLog(prog, 512, nullptr, buf);
        std::cerr << "Program link error:\n" << buf << std::endl;
        glDeleteProgram(prog);
        prog = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void initRenderer(size_t maxParticles) {
    const char* lineVS = R"glsl(
#version 450 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)glsl";
    const char* lineFS = R"glsl(
#version 450 core
out vec4 oCol;
void main() { oCol = vec4(1.0); }
)glsl";
    lineProg = compileShader(lineVS, lineFS);

    const char* ptVS = R"glsl(
#version 450 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos,1.0);
    gl_PointSize = 4.0;
}
)glsl";
    const char* ptFS = R"glsl(
#version 450 core
out vec4 oCol;
void main() { oCol = vec4(1.0,0.5,0.0,1.0); }
)glsl";
    pointProg = compileShader(ptVS, ptFS);

    float xmin = -1, xmax = 1, ymin = -1, ymax = 1, zmin = -1, zmax = 1;
    float verts[]{ xmin,ymin,zmin, xmax,ymin,zmin, xmax,ymax,zmin, xmin,ymax,zmin,
                   xmin,ymin,zmax, xmax,ymin,zmax, xmax,ymax,zmax, xmin,ymax,zmax };
    unsigned int idxs[]{ 0,1, 1,2, 2,3, 3,0,
                         4,5, 5,6, 6,7, 7,4,
                         0,4, 1,5, 2,6, 3,7 };
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glGenBuffers(1, &cubeEBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idxs), idxs, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);



    glGenVertexArrays(1, &ptsVAO);
    glGenBuffers(1, &ptsVBO);
    glBindVertexArray(ptsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, ptsVBO);
    glBufferData(GL_ARRAY_BUFFER,
        maxParticles * sizeof(glm::vec3),
        nullptr,
        GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glBindVertexArray(0);
}

void updateAndDraw(float /*t*/, const Particle* devP, size_t N) {
    std::vector<Particle> hostP(N);
    cudaMemcpy(hostP.data(), devP, N * sizeof(Particle), cudaMemcpyDeviceToHost);

    std::vector<glm::vec3> pts(N);
    for (size_t i = 0; i < N; ++i)
        pts[i] = glm::vec3(hostP[i].x, hostP[i].y, hostP[i].z);

    glBindBuffer(GL_ARRAY_BUFFER, ptsVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, N * sizeof(glm::vec3), pts.data());

    int w, h;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &w, &h);
    float aspect = float(w) / float(h);
    glm::mat4 P = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    glm::mat4 V = glm::lookAt(glm::vec3(3, 3, 3), glm::vec3(0), glm::vec3(0, 1, 0));
    glm::mat4 M = glm::mat4(1.0f);
    glm::mat4 MVP = P * V * M;



    glUseProgram(lineProg);
    glUniformMatrix4fv(glGetUniformLocation(lineProg, "uMVP"), 1, GL_FALSE, &MVP[0][0]);
    glBindVertexArray(cubeVAO);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);



    glUseProgram(pointProg);
    glUniformMatrix4fv(glGetUniformLocation(pointProg, "uMVP"), 1, GL_FALSE, &MVP[0][0]);
    glBindVertexArray(ptsVAO);
    glDrawArrays(GL_POINTS, 0, (GLsizei)N);
}

void cleanupRenderer() {
    glDeleteProgram(lineProg);
    glDeleteProgram(pointProg);
    glDeleteBuffers(1, &cubeVBO);
    glDeleteBuffers(1, &cubeEBO);
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &ptsVBO);
    glDeleteVertexArrays(1, &ptsVAO);
}
