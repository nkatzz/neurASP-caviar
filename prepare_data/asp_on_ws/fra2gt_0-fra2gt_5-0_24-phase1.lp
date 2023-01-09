
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

	orientation(fra2gt_0,161,0). coords(fra2gt_0,187,190,0).  happensAt(appear(fra2gt_0),0). happensAt(walking(fra2gt_0),0). orientation(fra2gt_5,150,0). coords(fra2gt_5,56,96,0).  happensAt(appear(fra2gt_5),0). happensAt(walking(fra2gt_5),0).
orientation(fra2gt_0,161,1). coords(fra2gt_0,187,183,1).  visible(fra2gt_0,1). happensAt(walking(fra2gt_0),1). orientation(fra2gt_5,150,1). coords(fra2gt_5,56,96,1).  visible(fra2gt_5,1). happensAt(walking(fra2gt_5),1).
orientation(fra2gt_0,159,2). coords(fra2gt_0,187,181,2).  visible(fra2gt_0,2). happensAt(walking(fra2gt_0),2). orientation(fra2gt_5,150,2). coords(fra2gt_5,55,97,2).  visible(fra2gt_5,2). happensAt(walking(fra2gt_5),2).
orientation(fra2gt_0,159,3). coords(fra2gt_0,185,178,3).  visible(fra2gt_0,3). happensAt(walking(fra2gt_0),3). orientation(fra2gt_5,150,3). coords(fra2gt_5,55,97,3).  visible(fra2gt_5,3). happensAt(walking(fra2gt_5),3).
orientation(fra2gt_0,159,4). coords(fra2gt_0,183,174,4).  visible(fra2gt_0,4). happensAt(walking(fra2gt_0),4). orientation(fra2gt_5,150,4). coords(fra2gt_5,55,98,4).  visible(fra2gt_5,4). happensAt(walking(fra2gt_5),4).
orientation(fra2gt_0,159,5). coords(fra2gt_0,181,171,5).  visible(fra2gt_0,5). happensAt(walking(fra2gt_0),5). orientation(fra2gt_5,150,5). coords(fra2gt_5,55,98,5).  visible(fra2gt_5,5). happensAt(walking(fra2gt_5),5).
orientation(fra2gt_0,159,6). coords(fra2gt_0,180,167,6).  visible(fra2gt_0,6). happensAt(walking(fra2gt_0),6). orientation(fra2gt_5,150,6). coords(fra2gt_5,55,98,6).  visible(fra2gt_5,6). happensAt(walking(fra2gt_5),6).
orientation(fra2gt_0,159,7). coords(fra2gt_0,179,165,7).  visible(fra2gt_0,7). happensAt(walking(fra2gt_0),7). orientation(fra2gt_5,150,7). coords(fra2gt_5,55,98,7).  visible(fra2gt_5,7). happensAt(walking(fra2gt_5),7).
orientation(fra2gt_0,159,8). coords(fra2gt_0,179,163,8).  visible(fra2gt_0,8). happensAt(walking(fra2gt_0),8). orientation(fra2gt_5,150,8). coords(fra2gt_5,54,99,8).  visible(fra2gt_5,8). happensAt(walking(fra2gt_5),8).
orientation(fra2gt_0,159,9). coords(fra2gt_0,179,161,9).  visible(fra2gt_0,9). happensAt(walking(fra2gt_0),9). orientation(fra2gt_5,150,9). coords(fra2gt_5,53,99,9).  visible(fra2gt_5,9). happensAt(walking(fra2gt_5),9).
orientation(fra2gt_0,159,10). coords(fra2gt_0,179,160,10).  visible(fra2gt_0,10). happensAt(walking(fra2gt_0),10). orientation(fra2gt_5,150,10). coords(fra2gt_5,53,99,10).  visible(fra2gt_5,10). happensAt(walking(fra2gt_5),10).
orientation(fra2gt_0,159,11). coords(fra2gt_0,179,158,11).  visible(fra2gt_0,11). happensAt(walking(fra2gt_0),11). orientation(fra2gt_5,150,11). coords(fra2gt_5,52,99,11).  visible(fra2gt_5,11). happensAt(walking(fra2gt_5),11).
orientation(fra2gt_0,159,12). coords(fra2gt_0,179,155,12).  visible(fra2gt_0,12). happensAt(walking(fra2gt_0),12). orientation(fra2gt_5,150,12). coords(fra2gt_5,51,99,12).  visible(fra2gt_5,12). happensAt(walking(fra2gt_5),12).
orientation(fra2gt_0,159,13). coords(fra2gt_0,178,154,13).  visible(fra2gt_0,13). happensAt(walking(fra2gt_0),13). orientation(fra2gt_5,150,13). coords(fra2gt_5,50,100,13).  visible(fra2gt_5,13). happensAt(walking(fra2gt_5),13).
orientation(fra2gt_0,159,14). coords(fra2gt_0,177,151,14).  visible(fra2gt_0,14). happensAt(walking(fra2gt_0),14). orientation(fra2gt_5,150,14). coords(fra2gt_5,50,100,14).  visible(fra2gt_5,14). happensAt(walking(fra2gt_5),14).
orientation(fra2gt_0,159,15). coords(fra2gt_0,177,149,15).  visible(fra2gt_0,15). happensAt(walking(fra2gt_0),15). orientation(fra2gt_5,150,15). coords(fra2gt_5,49,100,15).  visible(fra2gt_5,15). happensAt(walking(fra2gt_5),15).
orientation(fra2gt_0,159,16). coords(fra2gt_0,177,149,16).  visible(fra2gt_0,16). happensAt(walking(fra2gt_0),16). orientation(fra2gt_5,150,16). coords(fra2gt_5,48,100,16).  visible(fra2gt_5,16). happensAt(walking(fra2gt_5),16).
orientation(fra2gt_0,159,17). coords(fra2gt_0,176,147,17).  visible(fra2gt_0,17). happensAt(walking(fra2gt_0),17). orientation(fra2gt_5,150,17). coords(fra2gt_5,48,100,17).  visible(fra2gt_5,17). happensAt(walking(fra2gt_5),17).
orientation(fra2gt_0,159,18). coords(fra2gt_0,176,146,18).  visible(fra2gt_0,18). happensAt(walking(fra2gt_0),18). orientation(fra2gt_5,150,18). coords(fra2gt_5,48,100,18).  visible(fra2gt_5,18). happensAt(walking(fra2gt_5),18).
orientation(fra2gt_0,159,19). coords(fra2gt_0,176,143,19).  visible(fra2gt_0,19). happensAt(walking(fra2gt_0),19). orientation(fra2gt_5,150,19). coords(fra2gt_5,48,100,19).  visible(fra2gt_5,19). happensAt(walking(fra2gt_5),19).
orientation(fra2gt_0,159,20). coords(fra2gt_0,176,142,20).  visible(fra2gt_0,20). happensAt(walking(fra2gt_0),20). orientation(fra2gt_5,150,20). coords(fra2gt_5,48,101,20).  visible(fra2gt_5,20). happensAt(walking(fra2gt_5),20).
orientation(fra2gt_0,159,21). coords(fra2gt_0,176,141,21).  visible(fra2gt_0,21). happensAt(walking(fra2gt_0),21). orientation(fra2gt_5,150,21). coords(fra2gt_5,48,101,21).  visible(fra2gt_5,21). happensAt(walking(fra2gt_5),21).
orientation(fra2gt_0,159,22). coords(fra2gt_0,176,139,22).  visible(fra2gt_0,22). happensAt(walking(fra2gt_0),22). orientation(fra2gt_5,150,22). coords(fra2gt_5,48,101,22).  visible(fra2gt_5,22). happensAt(walking(fra2gt_5),22).
orientation(fra2gt_0,159,23). coords(fra2gt_0,176,139,23).  visible(fra2gt_0,23). happensAt(walking(fra2gt_0),23). orientation(fra2gt_5,150,23). coords(fra2gt_5,48,102,23).  visible(fra2gt_5,23). happensAt(walking(fra2gt_5),23).
orientation(fra2gt_0,159,24). coords(fra2gt_0,175,137,24).  visible(fra2gt_0,24). happensAt(walking(fra2gt_0),24). orientation(fra2gt_5,150,24). coords(fra2gt_5,48,102,24).  visible(fra2gt_5,24). happensAt(walking(fra2gt_5),24).
#show holdsAt/2.