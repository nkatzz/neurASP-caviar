
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

	orientation(lb2gt_1,130,0). coords(lb2gt_1,198,74,0).  happensAt(appear(lb2gt_1),0). happensAt(walking(lb2gt_1),0). orientation(lb2gt_2,147,0). coords(lb2gt_2,26,48,0).  happensAt(appear(lb2gt_2),0). happensAt(walking(lb2gt_2),0).
orientation(lb2gt_1,130,1). coords(lb2gt_1,198,73,1).  visible(lb2gt_1,1). happensAt(walking(lb2gt_1),1). orientation(lb2gt_2,147,1). coords(lb2gt_2,25,48,1).  visible(lb2gt_2,1). happensAt(walking(lb2gt_2),1).
orientation(lb2gt_1,130,2). coords(lb2gt_1,199,73,2).  visible(lb2gt_1,2). happensAt(walking(lb2gt_1),2). orientation(lb2gt_2,147,2). coords(lb2gt_2,25,48,2).  visible(lb2gt_2,2). happensAt(walking(lb2gt_2),2).
orientation(lb2gt_1,130,3). coords(lb2gt_1,202,72,3).  visible(lb2gt_1,3). happensAt(walking(lb2gt_1),3). orientation(lb2gt_2,147,3). coords(lb2gt_2,25,48,3).  visible(lb2gt_2,3). happensAt(walking(lb2gt_2),3).
orientation(lb2gt_1,130,4). coords(lb2gt_1,205,71,4).  visible(lb2gt_1,4). happensAt(walking(lb2gt_1),4). orientation(lb2gt_2,147,4). coords(lb2gt_2,25,48,4).  visible(lb2gt_2,4). happensAt(walking(lb2gt_2),4).
orientation(lb2gt_1,130,5). coords(lb2gt_1,206,71,5).  visible(lb2gt_1,5). happensAt(walking(lb2gt_1),5). orientation(lb2gt_2,147,5). coords(lb2gt_2,23,48,5).  visible(lb2gt_2,5). happensAt(walking(lb2gt_2),5).
orientation(lb2gt_1,130,6). coords(lb2gt_1,208,70,6).  visible(lb2gt_1,6). happensAt(walking(lb2gt_1),6). orientation(lb2gt_2,147,6). coords(lb2gt_2,22,49,6).  visible(lb2gt_2,6). happensAt(walking(lb2gt_2),6).
orientation(lb2gt_1,130,7). coords(lb2gt_1,210,69,7).  visible(lb2gt_1,7). happensAt(walking(lb2gt_1),7). orientation(lb2gt_2,147,7). coords(lb2gt_2,22,49,7).  visible(lb2gt_2,7). happensAt(walking(lb2gt_2),7).
orientation(lb2gt_1,130,8). coords(lb2gt_1,212,68,8).  visible(lb2gt_1,8). happensAt(walking(lb2gt_1),8). orientation(lb2gt_2,147,8). coords(lb2gt_2,22,49,8).  visible(lb2gt_2,8). happensAt(walking(lb2gt_2),8).
orientation(lb2gt_1,130,9). coords(lb2gt_1,212,67,9).  visible(lb2gt_1,9). happensAt(walking(lb2gt_1),9). orientation(lb2gt_2,147,9). coords(lb2gt_2,22,49,9).  visible(lb2gt_2,9). happensAt(walking(lb2gt_2),9).
orientation(lb2gt_1,130,10). coords(lb2gt_1,213,65,10).  visible(lb2gt_1,10). happensAt(walking(lb2gt_1),10). orientation(lb2gt_2,147,10). coords(lb2gt_2,22,49,10).  visible(lb2gt_2,10). happensAt(walking(lb2gt_2),10).
orientation(lb2gt_1,130,11). coords(lb2gt_1,213,63,11).  visible(lb2gt_1,11). happensAt(walking(lb2gt_1),11). orientation(lb2gt_2,147,11). coords(lb2gt_2,22,49,11).  visible(lb2gt_2,11). happensAt(walking(lb2gt_2),11).
orientation(lb2gt_1,130,12). coords(lb2gt_1,214,60,12).  visible(lb2gt_1,12). happensAt(walking(lb2gt_1),12). orientation(lb2gt_2,147,12). coords(lb2gt_2,22,49,12).  visible(lb2gt_2,12). happensAt(walking(lb2gt_2),12).
orientation(lb2gt_1,130,13). coords(lb2gt_1,214,58,13).  visible(lb2gt_1,13). happensAt(walking(lb2gt_1),13). orientation(lb2gt_2,147,13). coords(lb2gt_2,22,49,13).  visible(lb2gt_2,13). happensAt(walking(lb2gt_2),13).
orientation(lb2gt_1,130,14). coords(lb2gt_1,215,55,14).  visible(lb2gt_1,14). happensAt(walking(lb2gt_1),14). orientation(lb2gt_2,147,14). coords(lb2gt_2,22,49,14).  visible(lb2gt_2,14). happensAt(walking(lb2gt_2),14).
orientation(lb2gt_1,130,15). coords(lb2gt_1,217,53,15).  visible(lb2gt_1,15). happensAt(walking(lb2gt_1),15). orientation(lb2gt_2,147,15). coords(lb2gt_2,21,49,15).  visible(lb2gt_2,15). happensAt(walking(lb2gt_2),15).
orientation(lb2gt_1,130,16). coords(lb2gt_1,219,53,16).  visible(lb2gt_1,16). happensAt(walking(lb2gt_1),16). orientation(lb2gt_2,147,16). coords(lb2gt_2,21,49,16).  visible(lb2gt_2,16). happensAt(walking(lb2gt_2),16).
orientation(lb2gt_1,130,17). coords(lb2gt_1,221,52,17).  visible(lb2gt_1,17). happensAt(walking(lb2gt_1),17). orientation(lb2gt_2,147,17). coords(lb2gt_2,21,49,17).  visible(lb2gt_2,17). happensAt(walking(lb2gt_2),17).
orientation(lb2gt_1,130,18). coords(lb2gt_1,224,51,18).  visible(lb2gt_1,18). happensAt(walking(lb2gt_1),18). orientation(lb2gt_2,147,18). coords(lb2gt_2,21,49,18).  visible(lb2gt_2,18). happensAt(walking(lb2gt_2),18).
orientation(lb2gt_1,130,19). coords(lb2gt_1,226,51,19).  visible(lb2gt_1,19). happensAt(walking(lb2gt_1),19). orientation(lb2gt_2,147,19). coords(lb2gt_2,21,50,19).  visible(lb2gt_2,19). happensAt(walking(lb2gt_2),19).
orientation(lb2gt_1,130,20). coords(lb2gt_1,227,50,20).  visible(lb2gt_1,20). happensAt(walking(lb2gt_1),20). orientation(lb2gt_2,147,20). coords(lb2gt_2,21,50,20).  visible(lb2gt_2,20). happensAt(walking(lb2gt_2),20).
orientation(lb2gt_1,130,21). coords(lb2gt_1,228,49,21).  visible(lb2gt_1,21). happensAt(walking(lb2gt_1),21). orientation(lb2gt_2,147,21). coords(lb2gt_2,21,50,21).  visible(lb2gt_2,21). happensAt(walking(lb2gt_2),21).
orientation(lb2gt_1,124,22). coords(lb2gt_1,229,47,22).  visible(lb2gt_1,22). happensAt(walking(lb2gt_1),22). orientation(lb2gt_2,147,22). coords(lb2gt_2,21,50,22).  visible(lb2gt_2,22). happensAt(walking(lb2gt_2),22).
orientation(lb2gt_1,121,23). coords(lb2gt_1,230,46,23).  visible(lb2gt_1,23). happensAt(walking(lb2gt_1),23). orientation(lb2gt_2,147,23). coords(lb2gt_2,21,50,23).  visible(lb2gt_2,23). happensAt(walking(lb2gt_2),23).
orientation(lb2gt_1,117,24). coords(lb2gt_1,231,44,24).  visible(lb2gt_1,24). happensAt(walking(lb2gt_1),24). orientation(lb2gt_2,147,24). coords(lb2gt_2,21,50,24).  visible(lb2gt_2,24). happensAt(walking(lb2gt_2),24).
#show holdsAt/2.