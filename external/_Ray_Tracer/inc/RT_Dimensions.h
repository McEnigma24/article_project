#pragma once
#include "_preprocessor_.h"

#include <limits>

struct d1
{
    unit x;

public:
    GPU_LINE(__host__ __device__)
    d1(const unit& xx = -1) : x(xx) {}
    GPU_LINE(__host__) void print(const string& var_name) const;
};

struct d2 : public d1
{
    unit y;

public:
    GPU_LINE(__host__ __device__)
    d2(const unit& xx = -1, const unit& yy = -1) : d1(xx), y(yy) {}
    GPU_LINE(__host__) void print(const string& var_name) const;
};

struct d3 : public d2
{
    unit z;

public:
    GPU_LINE(__host__ __device__)
    d3(const unit& xx = -1, const unit& yy = -1, const unit& zz = -1) : d2(xx, yy), z(zz) {}

    GPU_LINE(__host__) void print(const string& var_name) const;
    void rotate_left_right(const d3& pivot, const unit& angleInRads)
    {
        unit s = sin(angleInRads);
        unit c = cos(angleInRads);

        unit cx = pivot.x;
        unit cz = pivot.z;

        x -= cx;
        z -= cz;

        // rotate point
        unit xnew = x * c - z * s;
        unit znew = x * s + z * c;

        // translate point back:
        x = xnew + cx;
        z = znew + cz;
    }
    void rotate_up_down(const d3& pivot, const unit& angleInRads)
    {
        unit s = sin(angleInRads);
        unit c = cos(angleInRads);

        unit cy = pivot.y;
        unit cz = pivot.z;

        // translate point back to origin:
        y -= cy;
        z -= cz;

        // rotate point
        unit xnew = y * c - z * s;
        unit znew = y * s + z * c;

        // translate point back:
        y = xnew + cy;
        z = znew + cz;
    }
    void rotate_twist(const d3& pivot, const unit& angleInRads)
    {
        unit s = sin(angleInRads);
        unit c = cos(angleInRads);

        unit cx = pivot.x;
        unit cy = pivot.y;

        // translate point back to origin:
        x -= cx;
        y -= cy;

        // rotate point
        unit xnew = x * c - y * s;
        unit ynew = x * s + y * c;

        // translate point back:
        x = xnew + cx;
        y = ynew + cy;
    }

    GPU_LINE(__host__ __device__)
    void normalize()
    {
        unit distance = d3::distance(*this);

        x /= distance;
        y /= distance;
        z /= distance;
    }

    GPU_LINE(__host__ __device__)
    void negate()
    {
        x = -x;
        y = -y;
        z = -z;
    }

    GPU_LINE(__host__ __device__)
    unit dot_prod(const d3& other) { return (this->x * other.x) + (this->y * other.y) + (this->z * other.z); }

    GPU_LINE(__host__ __device__)
    d3 operator+(const d3& other) const { return d3((this->x + other.x), (this->y + other.y), (this->z + other.z)); }

    GPU_LINE(__host__ __device__)
    d3 operator-(const d3& other) const { return d3((this->x - other.x), (this->y - other.y), (this->z - other.z)); }

    GPU_LINE(__host__ __device__)
    d3 operator*(const unit& mult) const { return d3((this->x * mult), (this->y * mult), (this->z * mult)); }

    GPU_LINE(__host__ __device__)
    d3& operator+=(const d3& other)
    {
        this->x += other.x;
        this->y += other.y;
        this->z += other.z;

        return *this;
    }

    GPU_LINE(__host__ __device__)
    d3& operator-=(const d3& other)
    {
        this->x -= other.x;
        this->y -= other.y;
        this->z -= other.z;

        return *this;
    }

    GPU_LINE(__host__ __device__)
    d3& operator*=(const unit& mult)
    {
        this->x *= mult;
        this->y *= mult;
        this->z *= mult;

        return *this;
    }

    GPU_LINE(__host__ __device__)
    d3& operator=(const d3& other)
    {
        THIS_OTHER(x);
        THIS_OTHER(y);
        THIS_OTHER(z);

        return *this;
    }

    GPU_LINE(__host__ __device__)
    static void assign(d3& main, const d3& other)
    {
        member_assign(main, other, x);
        member_assign(main, other, y);
        member_assign(main, other, z);
    }

    GPU_LINE(__host__ __device__)
    static unit distance(const d3& not_normalized_vector)
    {
        unit sum = (not_normalized_vector.x * not_normalized_vector.x) + (not_normalized_vector.y * not_normalized_vector.y) +
                   (not_normalized_vector.z * not_normalized_vector.z);

        return sqrt(sum);
    }

    GPU_LINE(__host__ __device__)
    static unit distance_between(const d3& obj1, const d3& obj2) { return distance(d3(obj1.x - obj2.x, obj1.y - obj2.y, obj1.z - obj2.z)); }
};