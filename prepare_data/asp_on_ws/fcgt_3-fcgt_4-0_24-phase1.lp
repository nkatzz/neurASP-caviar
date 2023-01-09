
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

	orientation(fcgt_3,159,0). coords(fcgt_3,103,163,0).  happensAt(appear(fcgt_3),0). happensAt(inactive(fcgt_3),0). orientation(fcgt_4,166,0). coords(fcgt_4,106,215,0).  happensAt(appear(fcgt_4),0). happensAt(walking(fcgt_4),0).
orientation(fcgt_3,159,1). coords(fcgt_3,103,163,1).  visible(fcgt_3,1). happensAt(inactive(fcgt_3),1). orientation(fcgt_4,166,1). coords(fcgt_4,106,215,1).  visible(fcgt_4,1). happensAt(walking(fcgt_4),1).
orientation(fcgt_3,159,2). coords(fcgt_3,103,163,2).  visible(fcgt_3,2). happensAt(inactive(fcgt_3),2). orientation(fcgt_4,166,2). coords(fcgt_4,106,215,2).  visible(fcgt_4,2). happensAt(walking(fcgt_4),2).
orientation(fcgt_3,159,3). coords(fcgt_3,103,163,3).  visible(fcgt_3,3). happensAt(inactive(fcgt_3),3). orientation(fcgt_4,166,3). coords(fcgt_4,106,215,3).  visible(fcgt_4,3). happensAt(walking(fcgt_4),3).
orientation(fcgt_3,159,4). coords(fcgt_3,103,163,4).  visible(fcgt_3,4). happensAt(inactive(fcgt_3),4). orientation(fcgt_4,166,4). coords(fcgt_4,106,215,4).  visible(fcgt_4,4). happensAt(walking(fcgt_4),4).
orientation(fcgt_3,159,5). coords(fcgt_3,103,163,5).  visible(fcgt_3,5). happensAt(inactive(fcgt_3),5). orientation(fcgt_4,166,5). coords(fcgt_4,106,215,5).  visible(fcgt_4,5). happensAt(walking(fcgt_4),5).
orientation(fcgt_3,159,6). coords(fcgt_3,103,163,6).  visible(fcgt_3,6). happensAt(inactive(fcgt_3),6). orientation(fcgt_4,166,6). coords(fcgt_4,106,215,6).  visible(fcgt_4,6). happensAt(walking(fcgt_4),6).
orientation(fcgt_3,159,7). coords(fcgt_3,103,163,7).  visible(fcgt_3,7). happensAt(inactive(fcgt_3),7). orientation(fcgt_4,166,7). coords(fcgt_4,105,215,7).  visible(fcgt_4,7). happensAt(walking(fcgt_4),7).
orientation(fcgt_3,159,8). coords(fcgt_3,103,163,8).  visible(fcgt_3,8). happensAt(inactive(fcgt_3),8). orientation(fcgt_4,166,8). coords(fcgt_4,105,215,8).  visible(fcgt_4,8). happensAt(walking(fcgt_4),8).
orientation(fcgt_3,159,9). coords(fcgt_3,103,163,9).  visible(fcgt_3,9). happensAt(inactive(fcgt_3),9). orientation(fcgt_4,166,9). coords(fcgt_4,105,215,9).  visible(fcgt_4,9). happensAt(walking(fcgt_4),9).
orientation(fcgt_3,159,10). coords(fcgt_3,103,163,10).  visible(fcgt_3,10). happensAt(inactive(fcgt_3),10). orientation(fcgt_4,166,10). coords(fcgt_4,105,215,10).  visible(fcgt_4,10). happensAt(walking(fcgt_4),10).
orientation(fcgt_3,159,11). coords(fcgt_3,103,163,11).  visible(fcgt_3,11). happensAt(inactive(fcgt_3),11). orientation(fcgt_4,166,11). coords(fcgt_4,105,215,11).  visible(fcgt_4,11). happensAt(walking(fcgt_4),11).
orientation(fcgt_3,159,12). coords(fcgt_3,103,163,12).  visible(fcgt_3,12). happensAt(inactive(fcgt_3),12). orientation(fcgt_4,166,12). coords(fcgt_4,105,215,12).  visible(fcgt_4,12). happensAt(walking(fcgt_4),12).
orientation(fcgt_3,159,13). coords(fcgt_3,103,163,13).  visible(fcgt_3,13). happensAt(inactive(fcgt_3),13). orientation(fcgt_4,166,13). coords(fcgt_4,105,215,13).  visible(fcgt_4,13). happensAt(walking(fcgt_4),13).
orientation(fcgt_3,159,14). coords(fcgt_3,103,163,14).  visible(fcgt_3,14). happensAt(inactive(fcgt_3),14). orientation(fcgt_4,166,14). coords(fcgt_4,105,214,14).  visible(fcgt_4,14). happensAt(walking(fcgt_4),14).
orientation(fcgt_3,159,15). coords(fcgt_3,103,163,15).  visible(fcgt_3,15). happensAt(inactive(fcgt_3),15). orientation(fcgt_4,166,15). coords(fcgt_4,105,214,15).  visible(fcgt_4,15). happensAt(walking(fcgt_4),15).
orientation(fcgt_3,159,16). coords(fcgt_3,103,163,16).  visible(fcgt_3,16). happensAt(inactive(fcgt_3),16). orientation(fcgt_4,166,16). coords(fcgt_4,104,214,16).  visible(fcgt_4,16). happensAt(walking(fcgt_4),16).
orientation(fcgt_3,159,17). coords(fcgt_3,103,163,17).  visible(fcgt_3,17). happensAt(inactive(fcgt_3),17). orientation(fcgt_4,166,17). coords(fcgt_4,104,214,17).  visible(fcgt_4,17). happensAt(walking(fcgt_4),17).
orientation(fcgt_3,159,18). coords(fcgt_3,103,163,18).  visible(fcgt_3,18). happensAt(inactive(fcgt_3),18). orientation(fcgt_4,166,18). coords(fcgt_4,104,214,18).  visible(fcgt_4,18). happensAt(walking(fcgt_4),18).
orientation(fcgt_3,159,19). coords(fcgt_3,103,163,19).  visible(fcgt_3,19). happensAt(inactive(fcgt_3),19). orientation(fcgt_4,166,19). coords(fcgt_4,104,214,19).  visible(fcgt_4,19). happensAt(walking(fcgt_4),19).
orientation(fcgt_3,159,20). coords(fcgt_3,103,163,20).  visible(fcgt_3,20). happensAt(inactive(fcgt_3),20). orientation(fcgt_4,166,20). coords(fcgt_4,104,214,20).  visible(fcgt_4,20). happensAt(walking(fcgt_4),20).
orientation(fcgt_3,159,21). coords(fcgt_3,103,163,21).  visible(fcgt_3,21). happensAt(inactive(fcgt_3),21). orientation(fcgt_4,166,21). coords(fcgt_4,104,214,21).  visible(fcgt_4,21). happensAt(walking(fcgt_4),21).
orientation(fcgt_3,159,22). coords(fcgt_3,103,163,22).  visible(fcgt_3,22). happensAt(inactive(fcgt_3),22). orientation(fcgt_4,166,22). coords(fcgt_4,103,214,22).  visible(fcgt_4,22). happensAt(walking(fcgt_4),22).
orientation(fcgt_3,159,23). coords(fcgt_3,103,163,23).  visible(fcgt_3,23). happensAt(inactive(fcgt_3),23). orientation(fcgt_4,166,23). coords(fcgt_4,103,214,23).  visible(fcgt_4,23). happensAt(walking(fcgt_4),23).
orientation(fcgt_3,159,24). coords(fcgt_3,103,163,24).  visible(fcgt_3,24). happensAt(inactive(fcgt_3),24). orientation(fcgt_4,166,24). coords(fcgt_4,103,214,24).  visible(fcgt_4,24). happensAt(walking(fcgt_4),24).
#show holdsAt/2.