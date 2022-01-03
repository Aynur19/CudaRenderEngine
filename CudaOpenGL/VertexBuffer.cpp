#include "VertexBuffer.h"

VertexBuffer::VertexBuffer()
{
	// Получение группы свободных идентификаторов.
	// Этот запрос выделяет 1 свободный идентификатор. 
	glGenBuffersARB(1, &id);

	ok = (glIsBufferARB(id) == GL_TRUE);
	target = 0;
}

VertexBuffer::~VertexBuffer()
{
	glDeleteBuffersARB(1, &id);
}

void VertexBuffer::bind(GLenum theTarget)
{
	glBindBufferARB(target = theTarget, id);
}

void VertexBuffer::unbind()
{
	glBindBufferARB(target, 0);
}

void VertexBuffer::setData(unsigned size, const void* ptr, GLenum usage)
{
	glBufferDataARB(target, size, ptr, usage);
}

void VertexBuffer::setSubData(unsigned offs, unsigned size, const void* ptr)
{
	glBufferSubDataARB(target, offs, size, ptr);
}

void VertexBuffer::getSubData(unsigned offs, unsigned size, void* ptr)
{
	glGetBufferSubDataARB(target, offs, size, ptr);
}

void* VertexBuffer::map(GLenum access)
{
	return glMapBufferARB(target, access);
}

bool VertexBuffer::unmap()
{
	return glUnmapBufferARB(target) == GL_TRUE;
}

bool VertexBuffer::isSupported()
{
	return isExtensionSupported("GL_ARB_vertex_buffer_object");
}