
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

	orientation(br1gt_0,174,0). coords(br1gt_0,287,277,0).  happensAt(appear(br1gt_0),0). happensAt(walking(br1gt_0),0). orientation(br1gt_2,158,0). coords(br1gt_2,84,180,0).  happensAt(appear(br1gt_2),0). happensAt(walking(br1gt_2),0).
orientation(br1gt_0,174,1). coords(br1gt_0,287,277,1).  visible(br1gt_0,1). happensAt(walking(br1gt_0),1). orientation(br1gt_2,158,1). coords(br1gt_2,84,181,1).  visible(br1gt_2,1). happensAt(walking(br1gt_2),1).
orientation(br1gt_0,174,2). coords(br1gt_0,286,276,2).  visible(br1gt_0,2). happensAt(walking(br1gt_0),2). orientation(br1gt_2,158,2). coords(br1gt_2,84,182,2).  visible(br1gt_2,2). happensAt(walking(br1gt_2),2).
orientation(br1gt_0,172,3). coords(br1gt_0,285,275,3).  visible(br1gt_0,3). happensAt(walking(br1gt_0),3). orientation(br1gt_2,158,3). coords(br1gt_2,84,184,3).  visible(br1gt_2,3). happensAt(walking(br1gt_2),3).
orientation(br1gt_0,172,4). coords(br1gt_0,284,273,4).  visible(br1gt_0,4). happensAt(walking(br1gt_0),4). orientation(br1gt_2,158,4). coords(br1gt_2,84,183,4).  visible(br1gt_2,4). happensAt(walking(br1gt_2),4).
orientation(br1gt_0,172,5). coords(br1gt_0,283,273,5).  visible(br1gt_0,5). happensAt(walking(br1gt_0),5). orientation(br1gt_2,160,5). coords(br1gt_2,84,183,5).  visible(br1gt_2,5). happensAt(walking(br1gt_2),5).
orientation(br1gt_0,172,6). coords(br1gt_0,282,271,6).  visible(br1gt_0,6). happensAt(walking(br1gt_0),6). orientation(br1gt_2,160,6). coords(br1gt_2,84,184,6).  visible(br1gt_2,6). happensAt(walking(br1gt_2),6).
orientation(br1gt_0,172,7). coords(br1gt_0,281,270,7).  visible(br1gt_0,7). happensAt(walking(br1gt_0),7). orientation(br1gt_2,160,7). coords(br1gt_2,84,184,7).  visible(br1gt_2,7). happensAt(walking(br1gt_2),7).
orientation(br1gt_0,172,8). coords(br1gt_0,278,269,8).  visible(br1gt_0,8). happensAt(walking(br1gt_0),8). orientation(br1gt_2,160,8). coords(br1gt_2,84,184,8).  visible(br1gt_2,8). happensAt(walking(br1gt_2),8).
orientation(br1gt_0,173,9). coords(br1gt_0,277,268,9).  visible(br1gt_0,9). happensAt(walking(br1gt_0),9). orientation(br1gt_2,160,9). coords(br1gt_2,84,185,9).  visible(br1gt_2,9). happensAt(walking(br1gt_2),9).
orientation(br1gt_0,173,10). coords(br1gt_0,274,266,10).  visible(br1gt_0,10). happensAt(walking(br1gt_0),10). orientation(br1gt_2,160,10). coords(br1gt_2,84,186,10).  visible(br1gt_2,10). happensAt(walking(br1gt_2),10).
orientation(br1gt_0,175,11). coords(br1gt_0,270,263,11).  visible(br1gt_0,11). happensAt(walking(br1gt_0),11). orientation(br1gt_2,160,11). coords(br1gt_2,84,187,11).  visible(br1gt_2,11). happensAt(walking(br1gt_2),11).
orientation(br1gt_0,175,12). coords(br1gt_0,267,262,12).  visible(br1gt_0,12). happensAt(walking(br1gt_0),12). orientation(br1gt_2,160,12). coords(br1gt_2,84,189,12).  visible(br1gt_2,12). happensAt(walking(br1gt_2),12).
orientation(br1gt_0,175,13). coords(br1gt_0,264,260,13).  visible(br1gt_0,13). happensAt(walking(br1gt_0),13). orientation(br1gt_2,160,13). coords(br1gt_2,84,191,13).  visible(br1gt_2,13). happensAt(walking(br1gt_2),13).
orientation(br1gt_0,6,14). coords(br1gt_0,260,257,14).  visible(br1gt_0,14). happensAt(walking(br1gt_0),14). orientation(br1gt_2,160,14). coords(br1gt_2,84,191,14).  visible(br1gt_2,14). happensAt(walking(br1gt_2),14).
orientation(br1gt_0,173,15). coords(br1gt_0,259,252,15).  visible(br1gt_0,15). happensAt(walking(br1gt_0),15). orientation(br1gt_2,160,15). coords(br1gt_2,84,193,15).  visible(br1gt_2,15). happensAt(walking(br1gt_2),15).
orientation(br1gt_0,169,16). coords(br1gt_0,258,249,16).  visible(br1gt_0,16). happensAt(walking(br1gt_0),16). orientation(br1gt_2,160,16). coords(br1gt_2,84,193,16).  visible(br1gt_2,16). happensAt(walking(br1gt_2),16).
orientation(br1gt_0,165,17). coords(br1gt_0,258,245,17).  visible(br1gt_0,17). happensAt(walking(br1gt_0),17). orientation(br1gt_2,160,17). coords(br1gt_2,84,194,17).  visible(br1gt_2,17). happensAt(walking(br1gt_2),17).
orientation(br1gt_0,164,18). coords(br1gt_0,256,242,18).  visible(br1gt_0,18). happensAt(walking(br1gt_0),18). orientation(br1gt_2,160,18). coords(br1gt_2,84,194,18).  visible(br1gt_2,18). happensAt(walking(br1gt_2),18).
orientation(br1gt_0,162,19). coords(br1gt_0,256,241,19).  visible(br1gt_0,19). happensAt(walking(br1gt_0),19). orientation(br1gt_2,160,19). coords(br1gt_2,84,195,19).  visible(br1gt_2,19). happensAt(walking(br1gt_2),19).
orientation(br1gt_0,159,20). coords(br1gt_0,255,241,20).  visible(br1gt_0,20). happensAt(walking(br1gt_0),20). orientation(br1gt_2,160,20). coords(br1gt_2,84,195,20).  visible(br1gt_2,20). happensAt(walking(br1gt_2),20).
orientation(br1gt_0,159,21). coords(br1gt_0,254,240,21).  visible(br1gt_0,21). happensAt(walking(br1gt_0),21). orientation(br1gt_2,160,21). coords(br1gt_2,84,195,21).  visible(br1gt_2,21). happensAt(walking(br1gt_2),21).
orientation(br1gt_0,164,22). coords(br1gt_0,249,239,22).  visible(br1gt_0,22). happensAt(walking(br1gt_0),22). orientation(br1gt_2,160,22). coords(br1gt_2,84,195,22).  visible(br1gt_2,22). happensAt(walking(br1gt_2),22).
orientation(br1gt_0,169,23). coords(br1gt_0,246,238,23).  visible(br1gt_0,23). happensAt(walking(br1gt_0),23). orientation(br1gt_2,160,23). coords(br1gt_2,84,196,23).  visible(br1gt_2,23). happensAt(walking(br1gt_2),23).
orientation(br1gt_0,174,24). coords(br1gt_0,243,234,24).  visible(br1gt_0,24). happensAt(walking(br1gt_0),24). orientation(br1gt_2,160,24). coords(br1gt_2,84,196,24).  visible(br1gt_2,24). happensAt(walking(br1gt_2),24).
#show holdsAt/2.