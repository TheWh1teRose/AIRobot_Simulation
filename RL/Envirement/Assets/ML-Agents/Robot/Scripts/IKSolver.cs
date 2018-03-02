using UnityEngine;
using System.Collections;

public static class IKSolver
{
    public static void Solve(
        bool debug,
        float targetWeight,
        float rotationWeight,
        float hintWeight,
        float minReach,
        float maxReach,
        Transform startLimb,
        Transform middleLimb,
        Transform endLimb,
        Vector3 limbForward,
        Vector3 targetPos,
        Vector3 hintPos,
        Quaternion targetRot)
    {
        if (!startLimb || !middleLimb || !endLimb) return;
        Vector3 delta_target = targetPos - startLimb.position;
        Vector3 delta_joint = middleLimb.position - startLimb.position;
        float angle;
        Vector3 axis;

        Quaternion startLimb_sr = startLimb.rotation;
        Quaternion middleLimb_sr = middleLimb.rotation;

        //Force target in reach
        if (minReach > 0f && delta_target.magnitude < minReach)
        {
            delta_target = delta_target.normalized * minReach;
        }
        if (maxReach > 0f && maxReach > minReach && delta_target.magnitude > maxReach)
        {
            delta_target = delta_target.normalized * maxReach;
        }
        targetPos = startLimb.position + delta_target;

        if (hintWeight != 0f)
        {
            //Apply hint roll
            Vector3 currentRight = Vector3.Cross(delta_target, startLimb.rotation * limbForward);
            Vector3 desiredRight = Vector3.Cross(delta_target, hintPos - startLimb.position);
            angle = Vector3.Angle(currentRight, desiredRight);
            axis = Vector3.Cross(currentRight, desiredRight);
            startLimb.Rotate(axis, angle, Space.World);

            if (hintWeight != 1f)
            {
                //Apply hint weight
                startLimb.rotation = Quaternion.Lerp(startLimb_sr, startLimb.rotation, hintWeight);
            }
            startLimb_sr = startLimb.rotation;
        }

        if (targetWeight != 0f)
        {
            //Force start limb to look at target
            angle = Vector3.Angle(delta_joint, delta_target);
            axis = Vector3.Cross(delta_joint, delta_target);
            startLimb.Rotate(axis, angle, Space.World);

            //Force middle limb to look at target
            delta_target = targetPos - middleLimb.position;
            delta_joint = endLimb.position - middleLimb.position;
            angle = Vector3.Angle(delta_joint, delta_target);
            axis = Vector3.Cross(delta_joint, delta_target);
            middleLimb.Rotate(axis, angle, Space.World);

            //Calculate triangle sides
            float a = Vector3.Distance(startLimb.position, middleLimb.position);
            float b = Vector3.Distance(middleLimb.position, endLimb.position);
            float c = Mathf.Min(Vector3.Distance(startLimb.position, targetPos), a + b - 0.000001f);

            //Bend start limb
            angle = LawOfCosToDegree(a, c, b);
            axis = Vector3.Cross(targetPos - startLimb.position, startLimb.rotation * limbForward);
            startLimb.Rotate(axis, angle, Space.World);

            //Bend middle limb
            delta_joint = endLimb.position - middleLimb.position;
            delta_target = targetPos - middleLimb.position;
            angle = Vector3.Angle(delta_joint, delta_target);
            axis = Vector3.Cross(delta_joint, delta_target);
            middleLimb.Rotate(axis, angle, Space.World);

            if (targetWeight != 1f)
            {
                //Apply ik weights
                startLimb.rotation = Quaternion.Lerp(startLimb_sr, startLimb.rotation, targetWeight);
                middleLimb.rotation = Quaternion.Lerp(middleLimb_sr, middleLimb.rotation, targetWeight);
            }
        }

        if (rotationWeight != 0f)
        {
            //Apply end limb rotation with weight
            endLimb.rotation = Quaternion.Lerp(endLimb.rotation, targetRot, rotationWeight);
        }


        if (debug)
        {
            Debug.DrawLine(startLimb.position, endLimb.position, Color.blue);
            Debug.DrawLine(startLimb.position, middleLimb.position, Color.red);
            Debug.DrawLine(middleLimb.position, endLimb.position, Color.green);
        }
    }

    private static float LawOfCosToDegree(float a, float b, float c)
    {
        float deg = Mathf.Acos(((c * c) - ((a * a) + (b * b))) / (-(2 * a * b))) * Mathf.Rad2Deg;
        return float.IsNaN(deg) ? 0f : deg;
    }
}
