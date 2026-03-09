from __future__ import annotations
import numpy as np


def trs_matrix(translation, quaternion, scale):
    mat_s = np.identity(4, dtype=np.float32)
    mat_s[0, 0] = scale[0]; mat_s[1, 1] = scale[1]; mat_s[2, 2] = scale[2]

    w, x, y, z = quaternion
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    mat_r = np.zeros((4, 4), dtype=np.float32)
    mat_r[0,0]=1-2*(yy+zz); mat_r[0,1]=2*(xy+wz);  mat_r[0,2]=2*(xz-wy)
    mat_r[1,0]=2*(xy-wz);   mat_r[1,1]=1-2*(xx+zz); mat_r[1,2]=2*(yz+wx)
    mat_r[2,0]=2*(xz+wy);   mat_r[2,1]=2*(yz-wx);   mat_r[2,2]=1-2*(xx+yy)
    mat_r[3,3]=1.0

    mat_t = np.identity(4, dtype=np.float32)
    mat_t[3,0]=translation[0]; mat_t[3,1]=translation[1]; mat_t[3,2]=translation[2]
    return mat_s @ mat_r @ mat_t


def euler_to_quaternion(pitch, yaw, roll):
    cy,sy = np.cos(yaw*0.5),np.sin(yaw*0.5)
    cp,sp = np.cos(pitch*0.5),np.sin(pitch*0.5)
    cr,sr = np.cos(roll*0.5),np.sin(roll*0.5)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy], dtype=np.float32)


def quaternion_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float32)
    n = np.linalg.norm(axis)
    if n < 1e-8: return np.array([1,0,0,0], dtype=np.float32)
    axis /= n; h = angle_rad*0.5
    return np.array([np.cos(h), axis[0]*np.sin(h), axis[1]*np.sin(h), axis[2]*np.sin(h)], dtype=np.float32)


def quaternion_multiply(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dtype=np.float32)


def quaternion_rotate_vector(q, v):
    w,x,y,z=q; uv=np.cross([x,y,z],v); uuv=np.cross([x,y,z],uv)
    return v + 2*(w*uv+uuv)


def view_matrix_from_transform(position, quaternion):
    w,x,y,z=quaternion
    xx,yy,zz=x*x,y*y,z*z; xy,xz,yz=x*y,x*z,y*z; wx,wy,wz=w*x,w*y,w*z
    mat_r=np.zeros((4,4),dtype=np.float32)
    mat_r[0,0]=1-2*(yy+zz); mat_r[0,1]=2*(xy+wz);  mat_r[0,2]=2*(xz-wy)
    mat_r[1,0]=2*(xy-wz);   mat_r[1,1]=1-2*(xx+zz); mat_r[1,2]=2*(yz+wx)
    mat_r[2,0]=2*(xz+wy);   mat_r[2,1]=2*(yz-wx);   mat_r[2,2]=1-2*(xx+yy)
    mat_r[3,3]=1.0
    rot_inv=mat_r[:3,:3].T
    trans_inv=-np.asarray(position,dtype=np.float32) @ rot_inv
    view=np.zeros((4,4),dtype=np.float32)
    view[:3,:3]=rot_inv; view[3,:3]=trans_inv; view[3,3]=1.0
    return view


def look_at_quaternion(eye, target, up):
    fwd=(np.asarray(target,np.float32)-np.asarray(eye,np.float32))
    fwd/=np.linalg.norm(fwd)
    right=np.cross(fwd,up); right/=np.linalg.norm(right)
    true_up=np.cross(right,fwd)
    mat=np.zeros((4,4),dtype=np.float32)
    mat[0,:3]=right; mat[1,:3]=true_up; mat[2,:3]=-fwd; mat[3,3]=1.0
    trace=mat[0,0]+mat[1,1]+mat[2,2]
    if trace>0:
        s=0.5/np.sqrt(trace+1); w=0.25/s
        x=(mat[2,1]-mat[1,2])*s; y=(mat[0,2]-mat[2,0])*s; z=(mat[1,0]-mat[0,1])*s
    elif mat[0,0]>mat[1,1] and mat[0,0]>mat[2,2]:
        s=2*np.sqrt(1+mat[0,0]-mat[1,1]-mat[2,2])
        w=(mat[2,1]-mat[1,2])/s; x=0.25*s; y=(mat[0,1]+mat[1,0])/s; z=(mat[0,2]+mat[2,0])/s
    elif mat[1,1]>mat[2,2]:
        s=2*np.sqrt(1+mat[1,1]-mat[0,0]-mat[2,2])
        w=(mat[0,2]-mat[2,0])/s; x=(mat[0,1]+mat[1,0])/s; y=0.25*s; z=(mat[1,2]+mat[2,1])/s
    else:
        s=2*np.sqrt(1+mat[2,2]-mat[0,0]-mat[1,1])
        w=(mat[1,0]-mat[0,1])/s; x=(mat[0,2]+mat[2,0])/s; y=(mat[1,2]+mat[2,1])/s; z=0.25*s
    q=np.array([w,x,y,z],dtype=np.float32); return q/np.linalg.norm(q)


def perspective_matrix(fov_y_rad, aspect, near, far):
    # Row-major (v @ M): w_clip = -z_view, encoded at M[2,3]. Constant at M[3,2].
    f=np.float32(1/np.tan(fov_y_rad*0.5))
    m=np.zeros((4,4),dtype=np.float32)
    m[0,0]=f/aspect; m[1,1]=f
    m[2,2]=far/(near-far)
    m[2,3]=-1.0                      # z_view -> w_clip  (was [3,2]: col-major mistake)
    m[3,2]=(far*near)/(near-far)     # constant offset   (was [2,3]: col-major mistake)
    return m


def clip_edge_homogeneous(a, b):
    t0,t1=0.0,1.0; delta=b-a
    planes=[
        (-(delta[0]+delta[3]), a[0]+a[3]),
        (-(delta[3]-delta[0]), a[3]-a[0]),
        (-(delta[1]+delta[3]), a[1]+a[3]),
        (-(delta[3]-delta[1]), a[3]-a[1]),
        (-delta[2],            a[2]),
        (-(delta[3]-delta[2]), a[3]-a[2]),
    ]
    for p,q in planes:
        if p==0:
            if q<0: return None
        else:
            r=q/p
            if p<0:
                if r>t1: return None
                if r>t0: t0=r
            else:
                if r<t0: return None
                if r<t1: t1=r
    return None if t0>t1 else (a+t0*delta, a+t1*delta)


def extract_frustum_planes(vp):
    # Row-major VP: v_clip = v @ VP, so clip components come from VP columns.
    # Gribb-Hartmann for row-major: use COLUMNS of VP (not rows).
    col0,col1,col2,col3=vp[:,0],vp[:,1],vp[:,2],vp[:,3]
    raw=[col3+col0, col3-col0, col3+col1, col3-col1, col2, col3-col2]
    planes=np.zeros((6,4),dtype=np.float32)
    for i,p in enumerate(raw):
        n=np.linalg.norm(p[:3])
        planes[i]=p/n if n>1e-8 else p
    return planes


def sphere_in_frustum(planes, centre, radius):
    c=np.asarray(centre,dtype=np.float32)
    for A,B,C,D in planes:
        if A*c[0]+B*c[1]+C*c[2]+D < -radius: return False
    return True


def ndc_to_screen(ndc, screen_width, screen_height):
    x=np.clip(ndc[0],-1,1); y=np.clip(ndc[1],-1,1)
    return float((x+1)*0.5*screen_width), float((1-y)*0.5*screen_height)


def unproject_ray(screen_x, screen_y, screen_w, screen_h, view_matrix, proj_matrix):
    ndc_x=(2.0*screen_x)/screen_w-1.0; ndc_y=1.0-(2.0*screen_y)/screen_h
    # Row-major: v_clip = v @ VP, so unproject via inv_vp @ clip_col or clip_row @ inv_vp
    clip=np.array([ndc_x,ndc_y,-1,1],dtype=np.float32)
    vp=view_matrix@proj_matrix           # row-major VP
    inv_vp=np.linalg.inv(vp)
    # Row-major unproject: world = clip @ inv_vp
    wp_h=clip@inv_vp; wp=wp_h[:3]/wp_h[3]
    # Camera origin: row 3 of view matrix holds -cam_pos @ rot_inv
    # So cam_pos = -view[3,:3] @ rot.T, where rot = view[:3,:3]
    rot=view_matrix[:3,:3]; trans=view_matrix[3,:3]
    origin=-trans@rot.T
    d=wp-origin; return origin.astype(np.float32), (d/np.linalg.norm(d)).astype(np.float32)


def ray_sphere_intersect(origin, direction, centre, radius):
    L=centre-origin; tca=np.dot(L,direction)
    if tca<0: return False
    return np.dot(L,L)-tca*tca <= radius*radius


def ray_segment_distance(ray_origin, ray_dir, seg_a, seg_b):
    u=seg_b-seg_a; v=ray_dir; w=ray_origin-seg_a
    a=np.dot(v,v); b=np.dot(v,u); c=np.dot(u,u); d=np.dot(v,w); e=np.dot(u,w)
    D=a*c-b*b
    t=(e/c if c>1e-8 else 0) if D<1e-8 else np.clip((a*e-b*d)/D,0,1)
    t=np.clip(t,0,1)
    csp=seg_a+t*u; s=max(np.dot(csp-ray_origin,v),0)
    return float(np.linalg.norm(ray_origin+s*v-(seg_a+t*u))), float(s)