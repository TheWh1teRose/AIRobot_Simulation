using UnityEngine;
using System.Collections;

[ExecuteInEditMode]
public class LimbIK : MonoBehaviour {
    
    public Vector3 elbowForward = Vector3.back;
    public Transform upperLimb;
    public Transform lowerLimb;
    public Transform endLimb;
    public Transform target;

	void LateUpdate ()
    {
        IKSolver.Solve(false, 1f, 1f, 0f, -1, -1, upperLimb, lowerLimb, endLimb, elbowForward, target.position, Vector3.zero, target.rotation);
	}
}
