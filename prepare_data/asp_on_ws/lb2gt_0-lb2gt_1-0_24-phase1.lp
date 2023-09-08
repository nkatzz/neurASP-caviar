
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

	orientation(lb2gt_0,141,0). coords(lb2gt_0,189,125,0).  happensAt(appear(lb2gt_0),0). happensAt(inactive(lb2gt_0),0). orientation(lb2gt_1,130,0). coords(lb2gt_1,198,74,0).  happensAt(appear(lb2gt_1),0). happensAt(walking(lb2gt_1),0).
orientation(lb2gt_0,141,1). coords(lb2gt_0,189,125,1).  visible(lb2gt_0,1). happensAt(inactive(lb2gt_0),1). orientation(lb2gt_1,130,1). coords(lb2gt_1,198,73,1).  visible(lb2gt_1,1). happensAt(walking(lb2gt_1),1).
orientation(lb2gt_0,141,2). coords(lb2gt_0,189,125,2).  visible(lb2gt_0,2). happensAt(inactive(lb2gt_0),2). orientation(lb2gt_1,130,2). coords(lb2gt_1,199,73,2).  visible(lb2gt_1,2). happensAt(walking(lb2gt_1),2).
orientation(lb2gt_0,141,3). coords(lb2gt_0,189,125,3).  visible(lb2gt_0,3). happensAt(inactive(lb2gt_0),3). orientation(lb2gt_1,130,3). coords(lb2gt_1,202,72,3).  visible(lb2gt_1,3). happensAt(walking(lb2gt_1),3).
orientation(lb2gt_0,141,4). coords(lb2gt_0,189,125,4).  visible(lb2gt_0,4). happensAt(inactive(lb2gt_0),4). orientation(lb2gt_1,130,4). coords(lb2gt_1,205,71,4).  visible(lb2gt_1,4). happensAt(walking(lb2gt_1),4).
orientation(lb2gt_0,141,5). coords(lb2gt_0,189,125,5).  visible(lb2gt_0,5). happensAt(inactive(lb2gt_0),5). orientation(lb2gt_1,130,5). coords(lb2gt_1,206,71,5).  visible(lb2gt_1,5). happensAt(walking(lb2gt_1),5).
orientation(lb2gt_0,141,6). coords(lb2gt_0,189,125,6).  visible(lb2gt_0,6). happensAt(inactive(lb2gt_0),6). orientation(lb2gt_1,130,6). coords(lb2gt_1,208,70,6).  visible(lb2gt_1,6). happensAt(walking(lb2gt_1),6).
orientation(lb2gt_0,141,7). coords(lb2gt_0,190,125,7).  visible(lb2gt_0,7). happensAt(inactive(lb2gt_0),7). orientation(lb2gt_1,130,7). coords(lb2gt_1,210,69,7).  visible(lb2gt_1,7). happensAt(walking(lb2gt_1),7).
orientation(lb2gt_0,141,8). coords(lb2gt_0,190,125,8).  visible(lb2gt_0,8). happensAt(inactive(lb2gt_0),8). orientation(lb2gt_1,130,8). coords(lb2gt_1,212,68,8).  visible(lb2gt_1,8). happensAt(walking(lb2gt_1),8).
orientation(lb2gt_0,141,9). coords(lb2gt_0,190,125,9).  visible(lb2gt_0,9). happensAt(inactive(lb2gt_0),9). orientation(lb2gt_1,130,9). coords(lb2gt_1,212,67,9).  visible(lb2gt_1,9). happensAt(walking(lb2gt_1),9).
orientation(lb2gt_0,141,10). coords(lb2gt_0,190,125,10).  visible(lb2gt_0,10). happensAt(inactive(lb2gt_0),10). orientation(lb2gt_1,130,10). coords(lb2gt_1,213,65,10).  visible(lb2gt_1,10). happensAt(walking(lb2gt_1),10).
orientation(lb2gt_0,141,11). coords(lb2gt_0,190,125,11).  visible(lb2gt_0,11). happensAt(inactive(lb2gt_0),11). orientation(lb2gt_1,130,11). coords(lb2gt_1,213,63,11).  visible(lb2gt_1,11). happensAt(walking(lb2gt_1),11).
orientation(lb2gt_0,141,12). coords(lb2gt_0,190,125,12).  visible(lb2gt_0,12). happensAt(inactive(lb2gt_0),12). orientation(lb2gt_1,130,12). coords(lb2gt_1,214,60,12).  visible(lb2gt_1,12). happensAt(walking(lb2gt_1),12).
orientation(lb2gt_0,141,13). coords(lb2gt_0,190,125,13).  visible(lb2gt_0,13). happensAt(inactive(lb2gt_0),13). orientation(lb2gt_1,130,13). coords(lb2gt_1,214,58,13).  visible(lb2gt_1,13). happensAt(walking(lb2gt_1),13).
orientation(lb2gt_0,141,14). coords(lb2gt_0,190,125,14).  visible(lb2gt_0,14). happensAt(inactive(lb2gt_0),14). orientation(lb2gt_1,130,14). coords(lb2gt_1,215,55,14).  visible(lb2gt_1,14). happensAt(walking(lb2gt_1),14).
orientation(lb2gt_0,141,15). coords(lb2gt_0,190,125,15).  visible(lb2gt_0,15). happensAt(inactive(lb2gt_0),15). orientation(lb2gt_1,130,15). coords(lb2gt_1,217,53,15).  visible(lb2gt_1,15). happensAt(walking(lb2gt_1),15).
orientation(lb2gt_0,141,16). coords(lb2gt_0,190,125,16).  visible(lb2gt_0,16). happensAt(inactive(lb2gt_0),16). orientation(lb2gt_1,130,16). coords(lb2gt_1,219,53,16).  visible(lb2gt_1,16). happensAt(walking(lb2gt_1),16).
orientation(lb2gt_0,141,17). coords(lb2gt_0,190,125,17).  visible(lb2gt_0,17). happensAt(inactive(lb2gt_0),17). orientation(lb2gt_1,130,17). coords(lb2gt_1,221,52,17).  visible(lb2gt_1,17). happensAt(walking(lb2gt_1),17).
orientation(lb2gt_0,141,18). coords(lb2gt_0,190,125,18).  visible(lb2gt_0,18). happensAt(inactive(lb2gt_0),18). orientation(lb2gt_1,130,18). coords(lb2gt_1,224,51,18).  visible(lb2gt_1,18). happensAt(walking(lb2gt_1),18).
orientation(lb2gt_0,141,19). coords(lb2gt_0,190,125,19).  visible(lb2gt_0,19). happensAt(inactive(lb2gt_0),19). orientation(lb2gt_1,130,19). coords(lb2gt_1,226,51,19).  visible(lb2gt_1,19). happensAt(walking(lb2gt_1),19).
orientation(lb2gt_0,141,20). coords(lb2gt_0,190,125,20).  visible(lb2gt_0,20). happensAt(inactive(lb2gt_0),20). orientation(lb2gt_1,130,20). coords(lb2gt_1,227,50,20).  visible(lb2gt_1,20). happensAt(walking(lb2gt_1),20).
orientation(lb2gt_0,141,21). coords(lb2gt_0,190,125,21).  visible(lb2gt_0,21). happensAt(inactive(lb2gt_0),21). orientation(lb2gt_1,130,21). coords(lb2gt_1,228,49,21).  visible(lb2gt_1,21). happensAt(walking(lb2gt_1),21).
orientation(lb2gt_0,141,22). coords(lb2gt_0,190,125,22).  visible(lb2gt_0,22). happensAt(inactive(lb2gt_0),22). orientation(lb2gt_1,124,22). coords(lb2gt_1,229,47,22).  visible(lb2gt_1,22). happensAt(walking(lb2gt_1),22).
orientation(lb2gt_0,141,23). coords(lb2gt_0,190,125,23).  visible(lb2gt_0,23). happensAt(inactive(lb2gt_0),23). orientation(lb2gt_1,121,23). coords(lb2gt_1,230,46,23).  visible(lb2gt_1,23). happensAt(walking(lb2gt_1),23).
orientation(lb2gt_0,141,24). coords(lb2gt_0,190,125,24).  visible(lb2gt_0,24). happensAt(inactive(lb2gt_0),24). orientation(lb2gt_1,117,24). coords(lb2gt_1,231,44,24).  visible(lb2gt_1,24). happensAt(walking(lb2gt_1),24).
#show holdsAt/2.