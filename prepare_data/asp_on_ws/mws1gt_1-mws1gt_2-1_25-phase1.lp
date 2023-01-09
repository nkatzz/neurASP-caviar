
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

	orientation(mws1gt_1,169,1). coords(mws1gt_1,65,193,1).  visible(mws1gt_1,1). happensAt(inactive(mws1gt_1),1). orientation(mws1gt_2,160,1). coords(mws1gt_2,46,102,1).  happensAt(appear(mws1gt_2),1). happensAt(walking(mws1gt_2),1).
orientation(mws1gt_1,169,2). coords(mws1gt_1,65,193,2).  visible(mws1gt_1,2). happensAt(inactive(mws1gt_1),2). orientation(mws1gt_2,160,2). coords(mws1gt_2,45,104,2).  visible(mws1gt_2,2). happensAt(walking(mws1gt_2),2).
orientation(mws1gt_1,170,3). coords(mws1gt_1,65,193,3).  visible(mws1gt_1,3). happensAt(inactive(mws1gt_1),3). orientation(mws1gt_2,160,3). coords(mws1gt_2,45,104,3).  visible(mws1gt_2,3). happensAt(walking(mws1gt_2),3).
orientation(mws1gt_1,170,4). coords(mws1gt_1,65,193,4).  visible(mws1gt_1,4). happensAt(inactive(mws1gt_1),4). orientation(mws1gt_2,0,4). coords(mws1gt_2,47,105,4).  visible(mws1gt_2,4). happensAt(walking(mws1gt_2),4).
orientation(mws1gt_1,170,5). coords(mws1gt_1,65,193,5).  visible(mws1gt_1,5). happensAt(inactive(mws1gt_1),5). orientation(mws1gt_2,0,5). coords(mws1gt_2,45,105,5).  visible(mws1gt_2,5). happensAt(walking(mws1gt_2),5).
orientation(mws1gt_1,170,6). coords(mws1gt_1,65,193,6).  visible(mws1gt_1,6). happensAt(inactive(mws1gt_1),6). orientation(mws1gt_2,0,6). coords(mws1gt_2,44,105,6).  visible(mws1gt_2,6). happensAt(walking(mws1gt_2),6).
orientation(mws1gt_1,170,7). coords(mws1gt_1,65,193,7).  visible(mws1gt_1,7). happensAt(inactive(mws1gt_1),7). orientation(mws1gt_2,142,7). coords(mws1gt_2,47,108,7).  visible(mws1gt_2,7). happensAt(walking(mws1gt_2),7).
orientation(mws1gt_1,170,8). coords(mws1gt_1,65,193,8).  visible(mws1gt_1,8). happensAt(inactive(mws1gt_1),8). orientation(mws1gt_2,144,8). coords(mws1gt_2,47,110,8).  visible(mws1gt_2,8). happensAt(walking(mws1gt_2),8).
orientation(mws1gt_1,170,9). coords(mws1gt_1,65,193,9).  visible(mws1gt_1,9). happensAt(inactive(mws1gt_1),9). orientation(mws1gt_2,144,9). coords(mws1gt_2,47,110,9).  visible(mws1gt_2,9). happensAt(walking(mws1gt_2),9).
orientation(mws1gt_1,170,10). coords(mws1gt_1,65,193,10).  visible(mws1gt_1,10). happensAt(inactive(mws1gt_1),10). orientation(mws1gt_2,144,10). coords(mws1gt_2,47,112,10).  visible(mws1gt_2,10). happensAt(walking(mws1gt_2),10).
orientation(mws1gt_1,170,11). coords(mws1gt_1,65,193,11).  visible(mws1gt_1,11). happensAt(inactive(mws1gt_1),11). orientation(mws1gt_2,144,11). coords(mws1gt_2,48,113,11).  visible(mws1gt_2,11). happensAt(walking(mws1gt_2),11).
orientation(mws1gt_1,170,12). coords(mws1gt_1,65,193,12).  visible(mws1gt_1,12). happensAt(inactive(mws1gt_1),12). orientation(mws1gt_2,144,12). coords(mws1gt_2,48,113,12).  visible(mws1gt_2,12). happensAt(walking(mws1gt_2),12).
orientation(mws1gt_1,170,13). coords(mws1gt_1,65,193,13).  visible(mws1gt_1,13). happensAt(inactive(mws1gt_1),13). orientation(mws1gt_2,144,13). coords(mws1gt_2,46,114,13).  visible(mws1gt_2,13). happensAt(walking(mws1gt_2),13).
orientation(mws1gt_1,170,14). coords(mws1gt_1,65,193,14).  visible(mws1gt_1,14). happensAt(inactive(mws1gt_1),14). orientation(mws1gt_2,144,14). coords(mws1gt_2,46,115,14).  visible(mws1gt_2,14). happensAt(walking(mws1gt_2),14).
orientation(mws1gt_1,170,15). coords(mws1gt_1,65,193,15).  visible(mws1gt_1,15). happensAt(inactive(mws1gt_1),15). orientation(mws1gt_2,144,15). coords(mws1gt_2,47,116,15).  visible(mws1gt_2,15). happensAt(walking(mws1gt_2),15).
orientation(mws1gt_1,170,16). coords(mws1gt_1,65,193,16).  visible(mws1gt_1,16). happensAt(inactive(mws1gt_1),16). orientation(mws1gt_2,144,16). coords(mws1gt_2,47,117,16).  visible(mws1gt_2,16). happensAt(walking(mws1gt_2),16).
orientation(mws1gt_1,170,17). coords(mws1gt_1,65,193,17).  visible(mws1gt_1,17). happensAt(inactive(mws1gt_1),17). orientation(mws1gt_2,144,17). coords(mws1gt_2,47,118,17).  visible(mws1gt_2,17). happensAt(walking(mws1gt_2),17).
orientation(mws1gt_1,170,18). coords(mws1gt_1,65,193,18).  visible(mws1gt_1,18). happensAt(inactive(mws1gt_1),18). orientation(mws1gt_2,155,18). coords(mws1gt_2,46,120,18).  visible(mws1gt_2,18). happensAt(walking(mws1gt_2),18).
orientation(mws1gt_1,170,19). coords(mws1gt_1,65,193,19).  visible(mws1gt_1,19). happensAt(inactive(mws1gt_1),19). orientation(mws1gt_2,155,19). coords(mws1gt_2,46,120,19).  visible(mws1gt_2,19). happensAt(walking(mws1gt_2),19).
orientation(mws1gt_1,170,20). coords(mws1gt_1,65,193,20).  visible(mws1gt_1,20). happensAt(inactive(mws1gt_1),20). orientation(mws1gt_2,155,20). coords(mws1gt_2,46,121,20).  visible(mws1gt_2,20). happensAt(walking(mws1gt_2),20).
orientation(mws1gt_1,170,21). coords(mws1gt_1,65,193,21).  visible(mws1gt_1,21). happensAt(inactive(mws1gt_1),21). orientation(mws1gt_2,155,21). coords(mws1gt_2,46,122,21).  visible(mws1gt_2,21). happensAt(walking(mws1gt_2),21).
orientation(mws1gt_1,170,22). coords(mws1gt_1,65,193,22).  visible(mws1gt_1,22). happensAt(inactive(mws1gt_1),22). orientation(mws1gt_2,155,22). coords(mws1gt_2,44,123,22).  visible(mws1gt_2,22). happensAt(walking(mws1gt_2),22).
orientation(mws1gt_1,170,23). coords(mws1gt_1,65,193,23).  visible(mws1gt_1,23). happensAt(inactive(mws1gt_1),23). orientation(mws1gt_2,155,23). coords(mws1gt_2,44,124,23).  visible(mws1gt_2,23). happensAt(walking(mws1gt_2),23).
orientation(mws1gt_1,170,24). coords(mws1gt_1,65,193,24).  visible(mws1gt_1,24). happensAt(inactive(mws1gt_1),24). orientation(mws1gt_2,155,24). coords(mws1gt_2,44,125,24).  visible(mws1gt_2,24). happensAt(walking(mws1gt_2),24).
orientation(mws1gt_1,170,25). coords(mws1gt_1,65,193,25).  visible(mws1gt_1,25). happensAt(inactive(mws1gt_1),25). orientation(mws1gt_2,155,25). coords(mws1gt_2,44,126,25).  visible(mws1gt_2,25). happensAt(walking(mws1gt_2),25).
#show holdsAt/2.