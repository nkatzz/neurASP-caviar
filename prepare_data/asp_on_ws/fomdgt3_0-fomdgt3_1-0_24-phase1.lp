
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

	orientation(fomdgt3_0,160,0). coords(fomdgt3_0,201,176,0).  happensAt(appear(fomdgt3_0),0). happensAt(walking(fomdgt3_0),0). orientation(fomdgt3_1,160,0). coords(fomdgt3_1,154,162,0).  happensAt(appear(fomdgt3_1),0). happensAt(active(fomdgt3_1),0).
orientation(fomdgt3_0,160,1). coords(fomdgt3_0,203,180,1).  visible(fomdgt3_0,1). happensAt(walking(fomdgt3_0),1). orientation(fomdgt3_1,160,1). coords(fomdgt3_1,154,162,1).  visible(fomdgt3_1,1). happensAt(active(fomdgt3_1),1).
orientation(fomdgt3_0,160,2). coords(fomdgt3_0,203,180,2).  visible(fomdgt3_0,2). happensAt(walking(fomdgt3_0),2). orientation(fomdgt3_1,160,2). coords(fomdgt3_1,154,162,2).  visible(fomdgt3_1,2). happensAt(active(fomdgt3_1),2).
orientation(fomdgt3_0,150,3). coords(fomdgt3_0,208,186,3).  visible(fomdgt3_0,3). happensAt(walking(fomdgt3_0),3). orientation(fomdgt3_1,160,3). coords(fomdgt3_1,154,162,3).  visible(fomdgt3_1,3). happensAt(active(fomdgt3_1),3).
orientation(fomdgt3_0,150,4). coords(fomdgt3_0,211,188,4).  visible(fomdgt3_0,4). happensAt(walking(fomdgt3_0),4). orientation(fomdgt3_1,160,4). coords(fomdgt3_1,154,162,4).  visible(fomdgt3_1,4). happensAt(active(fomdgt3_1),4).
orientation(fomdgt3_0,150,5). coords(fomdgt3_0,212,189,5).  visible(fomdgt3_0,5). happensAt(walking(fomdgt3_0),5). orientation(fomdgt3_1,160,5). coords(fomdgt3_1,154,162,5).  visible(fomdgt3_1,5). happensAt(active(fomdgt3_1),5).
orientation(fomdgt3_0,160,6). coords(fomdgt3_0,213,190,6).  visible(fomdgt3_0,6). happensAt(walking(fomdgt3_0),6). orientation(fomdgt3_1,160,6). coords(fomdgt3_1,154,162,6).  visible(fomdgt3_1,6). happensAt(active(fomdgt3_1),6).
orientation(fomdgt3_0,160,7). coords(fomdgt3_0,213,191,7).  visible(fomdgt3_0,7). happensAt(walking(fomdgt3_0),7). orientation(fomdgt3_1,160,7). coords(fomdgt3_1,154,162,7).  visible(fomdgt3_1,7). happensAt(active(fomdgt3_1),7).
orientation(fomdgt3_0,150,8). coords(fomdgt3_0,215,192,8).  visible(fomdgt3_0,8). happensAt(walking(fomdgt3_0),8). orientation(fomdgt3_1,160,8). coords(fomdgt3_1,154,162,8).  visible(fomdgt3_1,8). happensAt(active(fomdgt3_1),8).
orientation(fomdgt3_0,160,9). coords(fomdgt3_0,216,193,9).  visible(fomdgt3_0,9). happensAt(walking(fomdgt3_0),9). orientation(fomdgt3_1,160,9). coords(fomdgt3_1,154,162,9).  visible(fomdgt3_1,9). happensAt(active(fomdgt3_1),9).
orientation(fomdgt3_0,165,10). coords(fomdgt3_0,216,196,10).  visible(fomdgt3_0,10). happensAt(walking(fomdgt3_0),10). orientation(fomdgt3_1,160,10). coords(fomdgt3_1,154,162,10).  visible(fomdgt3_1,10). happensAt(active(fomdgt3_1),10).
orientation(fomdgt3_0,170,11). coords(fomdgt3_0,221,200,11).  visible(fomdgt3_0,11). happensAt(walking(fomdgt3_0),11). orientation(fomdgt3_1,160,11). coords(fomdgt3_1,154,162,11).  visible(fomdgt3_1,11). happensAt(active(fomdgt3_1),11).
orientation(fomdgt3_0,170,12). coords(fomdgt3_0,221,200,12).  visible(fomdgt3_0,12). happensAt(walking(fomdgt3_0),12). orientation(fomdgt3_1,160,12). coords(fomdgt3_1,154,162,12).  visible(fomdgt3_1,12). happensAt(active(fomdgt3_1),12).
orientation(fomdgt3_0,170,13). coords(fomdgt3_0,224,204,13).  visible(fomdgt3_0,13). happensAt(walking(fomdgt3_0),13). orientation(fomdgt3_1,160,13). coords(fomdgt3_1,154,162,13).  visible(fomdgt3_1,13). happensAt(active(fomdgt3_1),13).
orientation(fomdgt3_0,170,14). coords(fomdgt3_0,228,207,14).  visible(fomdgt3_0,14). happensAt(walking(fomdgt3_0),14). orientation(fomdgt3_1,160,14). coords(fomdgt3_1,155,162,14).  visible(fomdgt3_1,14). happensAt(active(fomdgt3_1),14).
orientation(fomdgt3_0,170,15). coords(fomdgt3_0,231,209,15).  visible(fomdgt3_0,15). happensAt(walking(fomdgt3_0),15). orientation(fomdgt3_1,160,15). coords(fomdgt3_1,155,162,15).  visible(fomdgt3_1,15). happensAt(active(fomdgt3_1),15).
orientation(fomdgt3_0,170,16). coords(fomdgt3_0,234,213,16).  visible(fomdgt3_0,16). happensAt(walking(fomdgt3_0),16). orientation(fomdgt3_1,160,16). coords(fomdgt3_1,155,162,16).  visible(fomdgt3_1,16). happensAt(active(fomdgt3_1),16).
orientation(fomdgt3_0,179,17). coords(fomdgt3_0,236,214,17).  visible(fomdgt3_0,17). happensAt(walking(fomdgt3_0),17). orientation(fomdgt3_1,160,17). coords(fomdgt3_1,155,162,17).  visible(fomdgt3_1,17). happensAt(active(fomdgt3_1),17).
orientation(fomdgt3_0,179,18). coords(fomdgt3_0,238,216,18).  visible(fomdgt3_0,18). happensAt(walking(fomdgt3_0),18). orientation(fomdgt3_1,160,18). coords(fomdgt3_1,155,162,18).  visible(fomdgt3_1,18). happensAt(active(fomdgt3_1),18).
orientation(fomdgt3_0,179,19). coords(fomdgt3_0,239,216,19).  visible(fomdgt3_0,19). happensAt(walking(fomdgt3_0),19). orientation(fomdgt3_1,160,19). coords(fomdgt3_1,155,162,19).  visible(fomdgt3_1,19). happensAt(active(fomdgt3_1),19).
orientation(fomdgt3_0,179,20). coords(fomdgt3_0,240,219,20).  visible(fomdgt3_0,20). happensAt(walking(fomdgt3_0),20). orientation(fomdgt3_1,160,20). coords(fomdgt3_1,155,162,20).  visible(fomdgt3_1,20). happensAt(active(fomdgt3_1),20).
orientation(fomdgt3_0,179,21). coords(fomdgt3_0,241,222,21).  visible(fomdgt3_0,21). happensAt(walking(fomdgt3_0),21). orientation(fomdgt3_1,160,21). coords(fomdgt3_1,155,162,21).  visible(fomdgt3_1,21). happensAt(active(fomdgt3_1),21).
orientation(fomdgt3_0,179,22). coords(fomdgt3_0,242,225,22).  visible(fomdgt3_0,22). happensAt(walking(fomdgt3_0),22). orientation(fomdgt3_1,160,22). coords(fomdgt3_1,155,162,22).  visible(fomdgt3_1,22). happensAt(active(fomdgt3_1),22).
orientation(fomdgt3_0,179,23). coords(fomdgt3_0,243,230,23).  visible(fomdgt3_0,23). happensAt(walking(fomdgt3_0),23). orientation(fomdgt3_1,160,23). coords(fomdgt3_1,155,162,23).  visible(fomdgt3_1,23). happensAt(active(fomdgt3_1),23).
orientation(fomdgt3_0,179,24). coords(fomdgt3_0,244,232,24).  visible(fomdgt3_0,24). happensAt(walking(fomdgt3_0),24). orientation(fomdgt3_1,160,24). coords(fomdgt3_1,155,162,24).  visible(fomdgt3_1,24). happensAt(active(fomdgt3_1),24).
#show holdsAt/2.