
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

	orientation(br1gt_0,174,0). coords(br1gt_0,287,277,0).  happensAt(appear(br1gt_0),0). happensAt(walking(br1gt_0),0). orientation(br1gt_1,173,0). coords(br1gt_1,210,242,0).  happensAt(appear(br1gt_1),0). happensAt(walking(br1gt_1),0).
orientation(br1gt_0,174,1). coords(br1gt_0,287,277,1).  visible(br1gt_0,1). happensAt(walking(br1gt_0),1). orientation(br1gt_1,173,1). coords(br1gt_1,209,241,1).  visible(br1gt_1,1). happensAt(walking(br1gt_1),1).
orientation(br1gt_0,174,2). coords(br1gt_0,286,276,2).  visible(br1gt_0,2). happensAt(walking(br1gt_0),2). orientation(br1gt_1,173,2). coords(br1gt_1,208,239,2).  visible(br1gt_1,2). happensAt(walking(br1gt_1),2).
orientation(br1gt_0,172,3). coords(br1gt_0,285,275,3).  visible(br1gt_0,3). happensAt(walking(br1gt_0),3). orientation(br1gt_1,173,3). coords(br1gt_1,206,238,3).  visible(br1gt_1,3). happensAt(walking(br1gt_1),3).
orientation(br1gt_0,172,4). coords(br1gt_0,284,273,4).  visible(br1gt_0,4). happensAt(walking(br1gt_0),4). orientation(br1gt_1,173,4). coords(br1gt_1,206,237,4).  visible(br1gt_1,4). happensAt(walking(br1gt_1),4).
orientation(br1gt_0,172,5). coords(br1gt_0,283,273,5).  visible(br1gt_0,5). happensAt(walking(br1gt_0),5). orientation(br1gt_1,173,5). coords(br1gt_1,203,234,5).  visible(br1gt_1,5). happensAt(walking(br1gt_1),5).
orientation(br1gt_0,172,6). coords(br1gt_0,282,271,6).  visible(br1gt_0,6). happensAt(walking(br1gt_0),6). orientation(br1gt_1,173,6). coords(br1gt_1,201,233,6).  visible(br1gt_1,6). happensAt(walking(br1gt_1),6).
orientation(br1gt_0,172,7). coords(br1gt_0,281,270,7).  visible(br1gt_0,7). happensAt(walking(br1gt_0),7). orientation(br1gt_1,173,7). coords(br1gt_1,199,233,7).  visible(br1gt_1,7). happensAt(walking(br1gt_1),7).
orientation(br1gt_0,172,8). coords(br1gt_0,278,269,8).  visible(br1gt_0,8). happensAt(walking(br1gt_0),8). orientation(br1gt_1,173,8). coords(br1gt_1,195,233,8).  visible(br1gt_1,8). happensAt(walking(br1gt_1),8).
orientation(br1gt_0,173,9). coords(br1gt_0,277,268,9).  visible(br1gt_0,9). happensAt(walking(br1gt_0),9). orientation(br1gt_1,173,9). coords(br1gt_1,191,231,9).  visible(br1gt_1,9). happensAt(walking(br1gt_1),9).
orientation(br1gt_0,173,10). coords(br1gt_0,274,266,10).  visible(br1gt_0,10). happensAt(walking(br1gt_0),10). orientation(br1gt_1,173,10). coords(br1gt_1,189,230,10).  visible(br1gt_1,10). happensAt(walking(br1gt_1),10).
orientation(br1gt_0,175,11). coords(br1gt_0,270,263,11).  visible(br1gt_0,11). happensAt(walking(br1gt_0),11). orientation(br1gt_1,173,11). coords(br1gt_1,187,229,11).  visible(br1gt_1,11). happensAt(walking(br1gt_1),11).
orientation(br1gt_0,175,12). coords(br1gt_0,267,262,12).  visible(br1gt_0,12). happensAt(walking(br1gt_0),12). orientation(br1gt_1,173,12). coords(br1gt_1,187,227,12).  visible(br1gt_1,12). happensAt(walking(br1gt_1),12).
orientation(br1gt_0,175,13). coords(br1gt_0,264,260,13).  visible(br1gt_0,13). happensAt(walking(br1gt_0),13). orientation(br1gt_1,173,13). coords(br1gt_1,186,225,13).  visible(br1gt_1,13). happensAt(walking(br1gt_1),13).
orientation(br1gt_0,6,14). coords(br1gt_0,260,257,14).  visible(br1gt_0,14). happensAt(walking(br1gt_0),14). orientation(br1gt_1,173,14). coords(br1gt_1,185,222,14).  visible(br1gt_1,14). happensAt(walking(br1gt_1),14).
orientation(br1gt_0,173,15). coords(br1gt_0,259,252,15).  visible(br1gt_0,15). happensAt(walking(br1gt_0),15). orientation(br1gt_1,173,15). coords(br1gt_1,185,221,15).  visible(br1gt_1,15). happensAt(walking(br1gt_1),15).
orientation(br1gt_0,169,16). coords(br1gt_0,258,249,16).  visible(br1gt_0,16). happensAt(walking(br1gt_0),16). orientation(br1gt_1,167,16). coords(br1gt_1,184,221,16).  visible(br1gt_1,16). happensAt(walking(br1gt_1),16).
orientation(br1gt_0,165,17). coords(br1gt_0,258,245,17).  visible(br1gt_0,17). happensAt(walking(br1gt_0),17). orientation(br1gt_1,167,17). coords(br1gt_1,182,220,17).  visible(br1gt_1,17). happensAt(walking(br1gt_1),17).
orientation(br1gt_0,164,18). coords(br1gt_0,256,242,18).  visible(br1gt_0,18). happensAt(walking(br1gt_0),18). orientation(br1gt_1,167,18). coords(br1gt_1,180,219,18).  visible(br1gt_1,18). happensAt(walking(br1gt_1),18).
orientation(br1gt_0,162,19). coords(br1gt_0,256,241,19).  visible(br1gt_0,19). happensAt(walking(br1gt_0),19). orientation(br1gt_1,167,19). coords(br1gt_1,178,219,19).  visible(br1gt_1,19). happensAt(walking(br1gt_1),19).
orientation(br1gt_0,159,20). coords(br1gt_0,255,241,20).  visible(br1gt_0,20). happensAt(walking(br1gt_0),20). orientation(br1gt_1,167,20). coords(br1gt_1,174,217,20).  visible(br1gt_1,20). happensAt(walking(br1gt_1),20).
orientation(br1gt_0,159,21). coords(br1gt_0,254,240,21).  visible(br1gt_0,21). happensAt(walking(br1gt_0),21). orientation(br1gt_1,167,21). coords(br1gt_1,172,215,21).  visible(br1gt_1,21). happensAt(walking(br1gt_1),21).
orientation(br1gt_0,164,22). coords(br1gt_0,249,239,22).  visible(br1gt_0,22). happensAt(walking(br1gt_0),22). orientation(br1gt_1,167,22). coords(br1gt_1,168,214,22).  visible(br1gt_1,22). happensAt(walking(br1gt_1),22).
orientation(br1gt_0,169,23). coords(br1gt_0,246,238,23).  visible(br1gt_0,23). happensAt(walking(br1gt_0),23). orientation(br1gt_1,167,23). coords(br1gt_1,166,213,23).  visible(br1gt_1,23). happensAt(walking(br1gt_1),23).
orientation(br1gt_0,174,24). coords(br1gt_0,243,234,24).  visible(br1gt_0,24). happensAt(walking(br1gt_0),24). orientation(br1gt_1,167,24). coords(br1gt_1,165,212,24).  visible(br1gt_1,24). happensAt(walking(br1gt_1),24).
#show holdsAt/2.