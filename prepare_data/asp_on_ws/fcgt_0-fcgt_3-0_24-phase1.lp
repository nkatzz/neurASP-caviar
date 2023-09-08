
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

	orientation(fcgt_0,154,0). coords(fcgt_0,133,124,0).  happensAt(appear(fcgt_0),0). happensAt(walking(fcgt_0),0). orientation(fcgt_3,159,0). coords(fcgt_3,103,163,0).  happensAt(appear(fcgt_3),0). happensAt(inactive(fcgt_3),0).
orientation(fcgt_0,154,1). coords(fcgt_0,135,124,1).  visible(fcgt_0,1). happensAt(walking(fcgt_0),1). orientation(fcgt_3,159,1). coords(fcgt_3,103,163,1).  visible(fcgt_3,1). happensAt(inactive(fcgt_3),1).
orientation(fcgt_0,158,2). coords(fcgt_0,138,123,2).  visible(fcgt_0,2). happensAt(walking(fcgt_0),2). orientation(fcgt_3,159,2). coords(fcgt_3,103,163,2).  visible(fcgt_3,2). happensAt(inactive(fcgt_3),2).
orientation(fcgt_0,158,3). coords(fcgt_0,141,122,3).  visible(fcgt_0,3). happensAt(walking(fcgt_0),3). orientation(fcgt_3,159,3). coords(fcgt_3,103,163,3).  visible(fcgt_3,3). happensAt(inactive(fcgt_3),3).
orientation(fcgt_0,158,4). coords(fcgt_0,143,122,4).  visible(fcgt_0,4). happensAt(walking(fcgt_0),4). orientation(fcgt_3,159,4). coords(fcgt_3,103,163,4).  visible(fcgt_3,4). happensAt(inactive(fcgt_3),4).
orientation(fcgt_0,163,5). coords(fcgt_0,143,121,5).  visible(fcgt_0,5). happensAt(walking(fcgt_0),5). orientation(fcgt_3,159,5). coords(fcgt_3,103,163,5).  visible(fcgt_3,5). happensAt(inactive(fcgt_3),5).
orientation(fcgt_0,166,6). coords(fcgt_0,146,120,6).  visible(fcgt_0,6). happensAt(walking(fcgt_0),6). orientation(fcgt_3,159,6). coords(fcgt_3,103,163,6).  visible(fcgt_3,6). happensAt(inactive(fcgt_3),6).
orientation(fcgt_0,165,7). coords(fcgt_0,147,119,7).  visible(fcgt_0,7). happensAt(walking(fcgt_0),7). orientation(fcgt_3,159,7). coords(fcgt_3,103,163,7).  visible(fcgt_3,7). happensAt(inactive(fcgt_3),7).
orientation(fcgt_0,160,8). coords(fcgt_0,148,119,8).  visible(fcgt_0,8). happensAt(walking(fcgt_0),8). orientation(fcgt_3,159,8). coords(fcgt_3,103,163,8).  visible(fcgt_3,8). happensAt(inactive(fcgt_3),8).
orientation(fcgt_0,155,9). coords(fcgt_0,148,119,9).  visible(fcgt_0,9). happensAt(walking(fcgt_0),9). orientation(fcgt_3,159,9). coords(fcgt_3,103,163,9).  visible(fcgt_3,9). happensAt(inactive(fcgt_3),9).
orientation(fcgt_0,154,10). coords(fcgt_0,149,119,10).  visible(fcgt_0,10). happensAt(walking(fcgt_0),10). orientation(fcgt_3,159,10). coords(fcgt_3,103,163,10).  visible(fcgt_3,10). happensAt(inactive(fcgt_3),10).
orientation(fcgt_0,150,11). coords(fcgt_0,149,119,11).  visible(fcgt_0,11). happensAt(walking(fcgt_0),11). orientation(fcgt_3,159,11). coords(fcgt_3,103,163,11).  visible(fcgt_3,11). happensAt(inactive(fcgt_3),11).
orientation(fcgt_0,150,12). coords(fcgt_0,149,119,12).  visible(fcgt_0,12). happensAt(walking(fcgt_0),12). orientation(fcgt_3,159,12). coords(fcgt_3,103,163,12).  visible(fcgt_3,12). happensAt(inactive(fcgt_3),12).
orientation(fcgt_0,150,13). coords(fcgt_0,150,119,13).  visible(fcgt_0,13). happensAt(walking(fcgt_0),13). orientation(fcgt_3,159,13). coords(fcgt_3,103,163,13).  visible(fcgt_3,13). happensAt(inactive(fcgt_3),13).
orientation(fcgt_0,150,14). coords(fcgt_0,151,121,14).  visible(fcgt_0,14). happensAt(walking(fcgt_0),14). orientation(fcgt_3,159,14). coords(fcgt_3,103,163,14).  visible(fcgt_3,14). happensAt(inactive(fcgt_3),14).
orientation(fcgt_0,150,15). coords(fcgt_0,152,122,15).  visible(fcgt_0,15). happensAt(walking(fcgt_0),15). orientation(fcgt_3,159,15). coords(fcgt_3,103,163,15).  visible(fcgt_3,15). happensAt(inactive(fcgt_3),15).
orientation(fcgt_0,150,16). coords(fcgt_0,155,122,16).  visible(fcgt_0,16). happensAt(walking(fcgt_0),16). orientation(fcgt_3,159,16). coords(fcgt_3,103,163,16).  visible(fcgt_3,16). happensAt(inactive(fcgt_3),16).
orientation(fcgt_0,150,17). coords(fcgt_0,158,123,17).  visible(fcgt_0,17). happensAt(walking(fcgt_0),17). orientation(fcgt_3,159,17). coords(fcgt_3,103,163,17).  visible(fcgt_3,17). happensAt(inactive(fcgt_3),17).
orientation(fcgt_0,150,18). coords(fcgt_0,159,123,18).  visible(fcgt_0,18). happensAt(walking(fcgt_0),18). orientation(fcgt_3,159,18). coords(fcgt_3,103,163,18).  visible(fcgt_3,18). happensAt(inactive(fcgt_3),18).
orientation(fcgt_0,150,19). coords(fcgt_0,161,123,19).  visible(fcgt_0,19). happensAt(walking(fcgt_0),19). orientation(fcgt_3,159,19). coords(fcgt_3,103,163,19).  visible(fcgt_3,19). happensAt(inactive(fcgt_3),19).
orientation(fcgt_0,150,20). coords(fcgt_0,163,123,20).  visible(fcgt_0,20). happensAt(walking(fcgt_0),20). orientation(fcgt_3,159,20). coords(fcgt_3,103,163,20).  visible(fcgt_3,20). happensAt(inactive(fcgt_3),20).
orientation(fcgt_0,150,21). coords(fcgt_0,164,123,21).  visible(fcgt_0,21). happensAt(walking(fcgt_0),21). orientation(fcgt_3,159,21). coords(fcgt_3,103,163,21).  visible(fcgt_3,21). happensAt(inactive(fcgt_3),21).
orientation(fcgt_0,150,22). coords(fcgt_0,164,123,22).  visible(fcgt_0,22). happensAt(walking(fcgt_0),22). orientation(fcgt_3,159,22). coords(fcgt_3,103,163,22).  visible(fcgt_3,22). happensAt(inactive(fcgt_3),22).
orientation(fcgt_0,150,23). coords(fcgt_0,165,123,23).  visible(fcgt_0,23). happensAt(walking(fcgt_0),23). orientation(fcgt_3,159,23). coords(fcgt_3,103,163,23).  visible(fcgt_3,23). happensAt(inactive(fcgt_3),23).
orientation(fcgt_0,150,24). coords(fcgt_0,165,123,24).  visible(fcgt_0,24). happensAt(walking(fcgt_0),24). orientation(fcgt_3,159,24). coords(fcgt_3,103,163,24).  visible(fcgt_3,24). happensAt(inactive(fcgt_3),24).
#show holdsAt/2.