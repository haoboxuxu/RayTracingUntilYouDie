//
//  bvh.h
//  RayTracingingTheNextWeek_cpp
//
//  Created by 徐浩博 on 2021/5/8.
//

#ifndef bvh_h
#define bvh_h

#include "utilitys.h"
#include "hittable.h"
#include "hittable_list.h"
#include <algorithm>

inline bool boxCompare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis);
bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b);
bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b);
bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b);

class BVHNode : public hittable {
public:
    BVHNode();
    BVHNode(const hittable_list& list, double time0, double time1)
            : BVHNode(list.objects, 0, list.objects.size(), time0, time1)
            {}
    BVHNode(const std::vector<shared_ptr<hittable>>& src_objects,
            size_t start, size_t end, double time0, double time1) {
        auto objects = src_objects;
        
        int axis = random_int(0, 2);
        auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;
        
        size_t object_span = end - start;
        
        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            if (comparator(objects[start], objects[start+1])) {
                left = objects[start];
                right = objects[start+1];
            } else {
                left = objects[start+1];
                right = objects[start];
            }
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);
            auto mid = start + object_span/2;
            left = make_shared<BVHNode>(objects, start, mid, time0, time1);
            right = make_shared<BVHNode>(objects, mid, end, time0, time1);
        }
    }
    
    bool hit(const Ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool boundingBox(double time0, double time1, AABB &output_box) const override;
    
public:
    shared_ptr<hittable> left;
    shared_ptr<hittable> right;
    AABB box;
};

bool BVHNode::boundingBox(double time0, double time1, AABB &output_box) const {
    output_box = box;
    return true;
}

bool BVHNode::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

inline bool boxCompare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis) {
    AABB box_a;
    AABB box_b;
    
    if (!a->boundingBox(0, 0, box_a) || !b->boundingBox(0, 0, box_b)) {
        std::cerr << "No bounding box in bvh_node constructor.\n";
    }
    
    return box_a.min().e[axis] < box_b.min().e[axis];
}

bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return boxCompare(a, b, 0);
}

bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return boxCompare(a, b, 1);
}

bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return boxCompare(a, b, 2);
}


#endif /* bvh_h */
