using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;

public class UnityPythonConector : MonoBehaviour {

    public GameObject traget;
    private UdpClient clientControls = null;
    private UdpClient clientPositionMatrix = null;
    float[,,] positionMatrix;

    float x = 35;
    float y = 35;
    float h = 11;
    int smothing = 8;
    //                         x,   y,    h
    float[] startPositions = {-1.3f,0.5f,1.3f};
    float[] endPositions   = {1.1f,2.9f,2f};

    // Use this for initialization
    void Start () {
        clientControls = new UdpClient(5002);
        clientPositionMatrix = new UdpClient(5003);

        positionMatrix = new float[Convert.ToInt32(x), Convert.ToInt32(y), Convert.ToInt32(h)];

    }
	
	// Update is called once per frame
	void Update () {
        UDPSendControls();
        updatePosiotionMatrix();
    }

    private void UDPSendControls()
    {
        string controls = Input.GetAxis("Horizontal") + ":"
            + -Input.GetAxis("Vertical") + ":"
            + -Input.GetAxis("Height") + ":"  + getGrab();

        byte[] positionByte = new byte[positionMatrix.Length * sizeof(float)];
        Buffer.BlockCopy(positionMatrix, 0, positionByte, 0, positionByte.Length);
        Debug.Log(positionByte.Length);

        IPEndPoint endpointControls = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5002);
        IPEndPoint endpointPositionMatrix = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5003);
        byte[] sendBytesControls = Encoding.ASCII.GetBytes(controls + "$");
        byte[] sendBytesPositionMatrix = new byte[positionMatrix.Length * sizeof(float)];
        Buffer.BlockCopy(positionMatrix, 0, sendBytesPositionMatrix, 0, sendBytesPositionMatrix.Length);
        clientControls.Send(sendBytesControls, sendBytesControls.Length, endpointControls);
        clientPositionMatrix.Send(sendBytesPositionMatrix, sendBytesPositionMatrix.Length, endpointPositionMatrix);
    }

    private void updatePosiotionMatrix()
    {
        //get the postions from the target. NOTE: some dimentions are switcht for realism
        float targetX = traget.transform.position.x;
        float targetY = traget.transform.position.z;
        float targetH = traget.transform.position.y;

        float distanceToStartX = targetX - startPositions[0];
        float distanceToStartY = targetY - startPositions[1];
        float distanceToStartH = targetH - startPositions[2];

        int positionInPosiotinMatrixX = Convert.ToInt32(Math.Round(distanceToStartX / ((endPositions[0] - startPositions[0]) / x)));
        int positionInPosiotinMatrixY = Convert.ToInt32(Math.Round(distanceToStartY / ((endPositions[1] - startPositions[1]) / y)));
        int positionInPosiotinMatrixH = Convert.ToInt32(Math.Round(distanceToStartH / ((endPositions[2] - startPositions[2]) / h)));

        Debug.Log(positionInPosiotinMatrixX + ":" + positionInPosiotinMatrixY + ":" + positionInPosiotinMatrixH);

        positionMatrix[positionInPosiotinMatrixX, positionInPosiotinMatrixY, positionInPosiotinMatrixH] = 1;

        for (int i = 0; i < (positionMatrix.GetLength(0)-1); i++)
        {
            for (int j = 0; j < (positionMatrix.GetLength(1)-1); j++)
            {
                for (int l = 0; l < (positionMatrix.GetLength(2)-1); l++)
                {
                    if (positionMatrix[i,j,l] > (Time.deltaTime/smothing))
                    {
                        positionMatrix[i, j, l] -= (Time.deltaTime / smothing);
                    }
                    else
                    {
                        positionMatrix[i, j, l] = 0;
                    }
                }
            }
        }

    }

    private int getGrab()
    {
        if (Input.GetKey("g"))
        {
            return 1;
        }else if (Input.GetKey("b"))
        {
            return -1;
        }
        return 0;
    }
}
