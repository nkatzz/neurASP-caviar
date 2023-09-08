
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

	orientation(wk2gt_0,3,0). coords(wk2gt_0,199,224,0).  happensAt(appear(wk2gt_0),0). happensAt(walking(wk2gt_0),0). orientation(wk2gt_1,143,0). coords(wk2gt_1,101,70,0).  happensAt(appear(wk2gt_1),0). happensAt(walking(wk2gt_1),0).
orientation(wk2gt_0,7,1). coords(wk2gt_0,201,226,1).  visible(wk2gt_0,1). happensAt(walking(wk2gt_0),1). orientation(wk2gt_1,143,1). coords(wk2gt_1,101,70,1).  visible(wk2gt_1,1). happensAt(walking(wk2gt_1),1).
orientation(wk2gt_0,7,2). coords(wk2gt_0,203,228,2).  visible(wk2gt_0,2). happensAt(walking(wk2gt_0),2). orientation(wk2gt_1,143,2). coords(wk2gt_1,102,70,2).  visible(wk2gt_1,2). happensAt(walking(wk2gt_1),2).
orientation(wk2gt_0,7,3). coords(wk2gt_0,204,230,3).  visible(wk2gt_0,3). happensAt(walking(wk2gt_0),3). orientation(wk2gt_1,143,3). coords(wk2gt_1,102,70,3).  visible(wk2gt_1,3). happensAt(walking(wk2gt_1),3).
orientation(wk2gt_0,12,4). coords(wk2gt_0,205,231,4).  visible(wk2gt_0,4). happensAt(walking(wk2gt_0),4). orientation(wk2gt_1,143,4). coords(wk2gt_1,102,69,4).  visible(wk2gt_1,4). happensAt(walking(wk2gt_1),4).
orientation(wk2gt_0,12,5). coords(wk2gt_0,206,232,5).  visible(wk2gt_0,5). happensAt(walking(wk2gt_0),5). orientation(wk2gt_1,143,5). coords(wk2gt_1,103,69,5).  visible(wk2gt_1,5). happensAt(walking(wk2gt_1),5).
orientation(wk2gt_0,12,6). coords(wk2gt_0,207,233,6).  visible(wk2gt_0,6). happensAt(walking(wk2gt_0),6). orientation(wk2gt_1,143,6). coords(wk2gt_1,102,68,6).  visible(wk2gt_1,6). happensAt(walking(wk2gt_1),6).
orientation(wk2gt_0,12,7). coords(wk2gt_0,207,234,7).  visible(wk2gt_0,7). happensAt(walking(wk2gt_0),7). orientation(wk2gt_1,143,7). coords(wk2gt_1,102,67,7).  visible(wk2gt_1,7). happensAt(walking(wk2gt_1),7).
orientation(wk2gt_0,12,8). coords(wk2gt_0,207,236,8).  visible(wk2gt_0,8). happensAt(walking(wk2gt_0),8). orientation(wk2gt_1,143,8). coords(wk2gt_1,103,67,8).  visible(wk2gt_1,8). happensAt(walking(wk2gt_1),8).
orientation(wk2gt_0,12,9). coords(wk2gt_0,208,238,9).  visible(wk2gt_0,9). happensAt(walking(wk2gt_0),9). orientation(wk2gt_1,143,9). coords(wk2gt_1,103,67,9).  visible(wk2gt_1,9). happensAt(walking(wk2gt_1),9).
orientation(wk2gt_0,12,10). coords(wk2gt_0,208,240,10).  visible(wk2gt_0,10). happensAt(walking(wk2gt_0),10). orientation(wk2gt_1,143,10). coords(wk2gt_1,103,66,10).  visible(wk2gt_1,10). happensAt(walking(wk2gt_1),10).
orientation(wk2gt_0,12,11). coords(wk2gt_0,209,242,11).  visible(wk2gt_0,11). happensAt(walking(wk2gt_0),11). orientation(wk2gt_1,143,11). coords(wk2gt_1,103,66,11).  visible(wk2gt_1,11). happensAt(walking(wk2gt_1),11).
orientation(wk2gt_0,12,12). coords(wk2gt_0,208,246,12).  visible(wk2gt_0,12). happensAt(walking(wk2gt_0),12). orientation(wk2gt_1,143,12). coords(wk2gt_1,103,65,12).  visible(wk2gt_1,12). happensAt(walking(wk2gt_1),12).
orientation(wk2gt_0,12,13). coords(wk2gt_0,208,248,13).  visible(wk2gt_0,13). happensAt(walking(wk2gt_0),13). orientation(wk2gt_1,143,13). coords(wk2gt_1,103,65,13).  visible(wk2gt_1,13). happensAt(walking(wk2gt_1),13).
orientation(wk2gt_0,12,14). coords(wk2gt_0,210,248,14).  visible(wk2gt_0,14). happensAt(walking(wk2gt_0),14). orientation(wk2gt_1,143,14). coords(wk2gt_1,104,65,14).  visible(wk2gt_1,14). happensAt(walking(wk2gt_1),14).
orientation(wk2gt_0,12,15). coords(wk2gt_0,210,251,15).  visible(wk2gt_0,15). happensAt(walking(wk2gt_0),15). orientation(wk2gt_1,143,15). coords(wk2gt_1,104,65,15).  visible(wk2gt_1,15). happensAt(walking(wk2gt_1),15).
orientation(wk2gt_0,12,16). coords(wk2gt_0,210,252,16).  visible(wk2gt_0,16). happensAt(walking(wk2gt_0),16). orientation(wk2gt_1,143,16). coords(wk2gt_1,104,65,16).  visible(wk2gt_1,16). happensAt(walking(wk2gt_1),16).
orientation(wk2gt_0,20,17). coords(wk2gt_0,210,253,17).  visible(wk2gt_0,17). happensAt(walking(wk2gt_0),17). orientation(wk2gt_1,143,17). coords(wk2gt_1,104,65,17).  visible(wk2gt_1,17). happensAt(walking(wk2gt_1),17).
orientation(wk2gt_0,20,18). coords(wk2gt_0,210,254,18).  visible(wk2gt_0,18). happensAt(walking(wk2gt_0),18). orientation(wk2gt_1,143,18). coords(wk2gt_1,105,64,18).  visible(wk2gt_1,18). happensAt(walking(wk2gt_1),18).
orientation(wk2gt_0,20,19). coords(wk2gt_0,210,256,19).  visible(wk2gt_0,19). happensAt(walking(wk2gt_0),19). orientation(wk2gt_1,143,19). coords(wk2gt_1,105,64,19).  visible(wk2gt_1,19). happensAt(walking(wk2gt_1),19).
orientation(wk2gt_0,31,20). coords(wk2gt_0,210,258,20).  visible(wk2gt_0,20). happensAt(walking(wk2gt_0),20). orientation(wk2gt_1,143,20). coords(wk2gt_1,105,64,20).  visible(wk2gt_1,20). happensAt(walking(wk2gt_1),20).
orientation(wk2gt_0,31,21). coords(wk2gt_0,211,261,21).  visible(wk2gt_0,21). happensAt(walking(wk2gt_0),21). orientation(wk2gt_1,143,21). coords(wk2gt_1,105,63,21).  visible(wk2gt_1,21). happensAt(walking(wk2gt_1),21).
orientation(wk2gt_0,31,22). coords(wk2gt_0,211,265,22).  visible(wk2gt_0,22). happensAt(walking(wk2gt_0),22). orientation(wk2gt_1,143,22). coords(wk2gt_1,105,62,22).  visible(wk2gt_1,22). happensAt(walking(wk2gt_1),22).
orientation(wk2gt_0,31,23). coords(wk2gt_0,212,268,23).  visible(wk2gt_0,23). happensAt(walking(wk2gt_0),23). orientation(wk2gt_1,143,23). coords(wk2gt_1,105,62,23).  visible(wk2gt_1,23). happensAt(walking(wk2gt_1),23).
orientation(wk2gt_0,31,24). coords(wk2gt_0,213,271,24).  visible(wk2gt_0,24). happensAt(walking(wk2gt_0),24). orientation(wk2gt_1,143,24). coords(wk2gt_1,105,62,24).  visible(wk2gt_1,24). happensAt(walking(wk2gt_1),24).
#show holdsAt/2.