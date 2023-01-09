
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lb1gt_0,165,0). coords(lb1gt_0,184,204,0).  happensAt(appear(lb1gt_0),0). happensAt(walking(lb1gt_0),0). orientation(lb1gt_2,142,0). coords(lb1gt_2,78,63,0).  happensAt(appear(lb1gt_2),0). happensAt(walking(lb1gt_2),0).
orientation(lb1gt_0,165,1). coords(lb1gt_0,183,200,1).  visible(lb1gt_0,1). happensAt(walking(lb1gt_0),1). orientation(lb1gt_2,142,1). coords(lb1gt_2,78,63,1).  visible(lb1gt_2,1). happensAt(walking(lb1gt_2),1).
orientation(lb1gt_0,165,2). coords(lb1gt_0,182,199,2).  visible(lb1gt_0,2). happensAt(walking(lb1gt_0),2). orientation(lb1gt_2,142,2). coords(lb1gt_2,78,63,2).  visible(lb1gt_2,2). happensAt(walking(lb1gt_2),2).
orientation(lb1gt_0,165,3). coords(lb1gt_0,182,197,3).  visible(lb1gt_0,3). happensAt(walking(lb1gt_0),3). orientation(lb1gt_2,142,3). coords(lb1gt_2,78,63,3).  visible(lb1gt_2,3). happensAt(walking(lb1gt_2),3).
orientation(lb1gt_0,165,4). coords(lb1gt_0,180,196,4).  visible(lb1gt_0,4). happensAt(walking(lb1gt_0),4). orientation(lb1gt_2,142,4). coords(lb1gt_2,78,63,4).  visible(lb1gt_2,4). happensAt(walking(lb1gt_2),4).
orientation(lb1gt_0,165,5). coords(lb1gt_0,180,193,5).  visible(lb1gt_0,5). happensAt(walking(lb1gt_0),5). orientation(lb1gt_2,142,5). coords(lb1gt_2,78,64,5).  visible(lb1gt_2,5). happensAt(walking(lb1gt_2),5).
orientation(lb1gt_0,159,6). coords(lb1gt_0,179,192,6).  visible(lb1gt_0,6). happensAt(walking(lb1gt_0),6). orientation(lb1gt_2,142,6). coords(lb1gt_2,79,64,6).  visible(lb1gt_2,6). happensAt(walking(lb1gt_2),6).
orientation(lb1gt_0,155,7). coords(lb1gt_0,179,191,7).  visible(lb1gt_0,7). happensAt(walking(lb1gt_0),7). orientation(lb1gt_2,142,7). coords(lb1gt_2,79,64,7).  visible(lb1gt_2,7). happensAt(walking(lb1gt_2),7).
orientation(lb1gt_0,155,8). coords(lb1gt_0,179,189,8).  visible(lb1gt_0,8). happensAt(walking(lb1gt_0),8). orientation(lb1gt_2,142,8). coords(lb1gt_2,79,64,8).  visible(lb1gt_2,8). happensAt(walking(lb1gt_2),8).
orientation(lb1gt_0,155,9). coords(lb1gt_0,178,185,9).  visible(lb1gt_0,9). happensAt(walking(lb1gt_0),9). orientation(lb1gt_2,142,9). coords(lb1gt_2,79,65,9).  visible(lb1gt_2,9). happensAt(walking(lb1gt_2),9).
orientation(lb1gt_0,155,10). coords(lb1gt_0,177,181,10).  visible(lb1gt_0,10). happensAt(walking(lb1gt_0),10). orientation(lb1gt_2,142,10). coords(lb1gt_2,79,65,10).  visible(lb1gt_2,10). happensAt(walking(lb1gt_2),10).
orientation(lb1gt_0,155,11). coords(lb1gt_0,176,177,11).  visible(lb1gt_0,11). happensAt(walking(lb1gt_0),11). orientation(lb1gt_2,142,11). coords(lb1gt_2,79,65,11).  visible(lb1gt_2,11). happensAt(walking(lb1gt_2),11).
orientation(lb1gt_0,155,12). coords(lb1gt_0,175,172,12).  visible(lb1gt_0,12). happensAt(walking(lb1gt_0),12). orientation(lb1gt_2,142,12). coords(lb1gt_2,80,66,12).  visible(lb1gt_2,12). happensAt(walking(lb1gt_2),12).
orientation(lb1gt_0,155,13). coords(lb1gt_0,174,168,13).  visible(lb1gt_0,13). happensAt(walking(lb1gt_0),13). orientation(lb1gt_2,142,13). coords(lb1gt_2,80,66,13).  visible(lb1gt_2,13). happensAt(walking(lb1gt_2),13).
orientation(lb1gt_0,155,14). coords(lb1gt_0,174,167,14).  visible(lb1gt_0,14). happensAt(walking(lb1gt_0),14). orientation(lb1gt_2,142,14). coords(lb1gt_2,80,66,14).  visible(lb1gt_2,14). happensAt(walking(lb1gt_2),14).
orientation(lb1gt_0,155,15). coords(lb1gt_0,174,165,15).  visible(lb1gt_0,15). happensAt(walking(lb1gt_0),15). orientation(lb1gt_2,142,15). coords(lb1gt_2,80,66,15).  visible(lb1gt_2,15). happensAt(walking(lb1gt_2),15).
orientation(lb1gt_0,150,16). coords(lb1gt_0,174,164,16).  visible(lb1gt_0,16). happensAt(walking(lb1gt_0),16). orientation(lb1gt_2,142,16). coords(lb1gt_2,80,66,16).  visible(lb1gt_2,16). happensAt(walking(lb1gt_2),16).
orientation(lb1gt_0,146,17). coords(lb1gt_0,174,164,17).  visible(lb1gt_0,17). happensAt(walking(lb1gt_0),17). orientation(lb1gt_2,142,17). coords(lb1gt_2,80,67,17).  visible(lb1gt_2,17). happensAt(walking(lb1gt_2),17).
orientation(lb1gt_0,144,18). coords(lb1gt_0,173,161,18).  visible(lb1gt_0,18). happensAt(walking(lb1gt_0),18). orientation(lb1gt_2,142,18). coords(lb1gt_2,80,67,18).  visible(lb1gt_2,18). happensAt(walking(lb1gt_2),18).
orientation(lb1gt_0,146,19). coords(lb1gt_0,173,160,19).  visible(lb1gt_0,19). happensAt(walking(lb1gt_0),19). orientation(lb1gt_2,142,19). coords(lb1gt_2,80,67,19).  visible(lb1gt_2,19). happensAt(walking(lb1gt_2),19).
orientation(lb1gt_0,146,20). coords(lb1gt_0,172,156,20).  visible(lb1gt_0,20). happensAt(walking(lb1gt_0),20). orientation(lb1gt_2,142,20). coords(lb1gt_2,80,68,20).  visible(lb1gt_2,20). happensAt(walking(lb1gt_2),20).
orientation(lb1gt_0,146,21). coords(lb1gt_0,171,153,21).  visible(lb1gt_0,21). happensAt(walking(lb1gt_0),21). orientation(lb1gt_2,142,21). coords(lb1gt_2,80,69,21).  visible(lb1gt_2,21). happensAt(walking(lb1gt_2),21).
orientation(lb1gt_0,146,22). coords(lb1gt_0,171,150,22).  visible(lb1gt_0,22). happensAt(walking(lb1gt_0),22). orientation(lb1gt_2,142,22). coords(lb1gt_2,80,70,22).  visible(lb1gt_2,22). happensAt(walking(lb1gt_2),22).
orientation(lb1gt_0,146,23). coords(lb1gt_0,171,149,23).  visible(lb1gt_0,23). happensAt(walking(lb1gt_0),23). orientation(lb1gt_2,142,23). coords(lb1gt_2,80,71,23).  visible(lb1gt_2,23). happensAt(walking(lb1gt_2),23).
orientation(lb1gt_0,146,24). coords(lb1gt_0,171,148,24).  visible(lb1gt_0,24). happensAt(walking(lb1gt_0),24). orientation(lb1gt_2,142,24). coords(lb1gt_2,80,71,24).  visible(lb1gt_2,24). happensAt(walking(lb1gt_2),24).
#show holdsAt/2.