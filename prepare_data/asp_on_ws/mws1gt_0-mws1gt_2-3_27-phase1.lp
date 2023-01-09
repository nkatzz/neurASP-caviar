
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

	orientation(mws1gt_0,170,3). coords(mws1gt_0,253,217,3).  visible(mws1gt_0,3). happensAt(walking(mws1gt_0),3). orientation(mws1gt_2,160,3). coords(mws1gt_2,45,104,3).  visible(mws1gt_2,3). happensAt(walking(mws1gt_2),3).
orientation(mws1gt_0,170,4). coords(mws1gt_0,257,219,4).  visible(mws1gt_0,4). happensAt(walking(mws1gt_0),4). orientation(mws1gt_2,0,4). coords(mws1gt_2,47,105,4).  visible(mws1gt_2,4). happensAt(walking(mws1gt_2),4).
orientation(mws1gt_0,175,5). coords(mws1gt_0,260,226,5).  visible(mws1gt_0,5). happensAt(walking(mws1gt_0),5). orientation(mws1gt_2,0,5). coords(mws1gt_2,45,105,5).  visible(mws1gt_2,5). happensAt(walking(mws1gt_2),5).
orientation(mws1gt_0,175,6). coords(mws1gt_0,260,229,6).  visible(mws1gt_0,6). happensAt(walking(mws1gt_0),6). orientation(mws1gt_2,0,6). coords(mws1gt_2,44,105,6).  visible(mws1gt_2,6). happensAt(walking(mws1gt_2),6).
orientation(mws1gt_0,175,7). coords(mws1gt_0,261,230,7).  visible(mws1gt_0,7). happensAt(walking(mws1gt_0),7). orientation(mws1gt_2,142,7). coords(mws1gt_2,47,108,7).  visible(mws1gt_2,7). happensAt(walking(mws1gt_2),7).
orientation(mws1gt_0,175,8). coords(mws1gt_0,263,233,8).  visible(mws1gt_0,8). happensAt(walking(mws1gt_0),8). orientation(mws1gt_2,144,8). coords(mws1gt_2,47,110,8).  visible(mws1gt_2,8). happensAt(walking(mws1gt_2),8).
orientation(mws1gt_0,19,9). coords(mws1gt_0,265,237,9).  visible(mws1gt_0,9). happensAt(walking(mws1gt_0),9). orientation(mws1gt_2,144,9). coords(mws1gt_2,47,110,9).  visible(mws1gt_2,9). happensAt(walking(mws1gt_2),9).
orientation(mws1gt_0,28,10). coords(mws1gt_0,266,239,10).  visible(mws1gt_0,10). happensAt(walking(mws1gt_0),10). orientation(mws1gt_2,144,10). coords(mws1gt_2,47,112,10).  visible(mws1gt_2,10). happensAt(walking(mws1gt_2),10).
orientation(mws1gt_0,28,11). coords(mws1gt_0,268,241,11).  visible(mws1gt_0,11). happensAt(walking(mws1gt_0),11). orientation(mws1gt_2,144,11). coords(mws1gt_2,48,113,11).  visible(mws1gt_2,11). happensAt(walking(mws1gt_2),11).
orientation(mws1gt_0,28,12). coords(mws1gt_0,269,244,12).  visible(mws1gt_0,12). happensAt(walking(mws1gt_0),12). orientation(mws1gt_2,144,12). coords(mws1gt_2,48,113,12).  visible(mws1gt_2,12). happensAt(walking(mws1gt_2),12).
orientation(mws1gt_0,28,13). coords(mws1gt_0,271,248,13).  visible(mws1gt_0,13). happensAt(walking(mws1gt_0),13). orientation(mws1gt_2,144,13). coords(mws1gt_2,46,114,13).  visible(mws1gt_2,13). happensAt(walking(mws1gt_2),13).
orientation(mws1gt_0,28,14). coords(mws1gt_0,274,253,14).  visible(mws1gt_0,14). happensAt(walking(mws1gt_0),14). orientation(mws1gt_2,144,14). coords(mws1gt_2,46,115,14).  visible(mws1gt_2,14). happensAt(walking(mws1gt_2),14).
orientation(mws1gt_0,28,15). coords(mws1gt_0,275,258,15).  visible(mws1gt_0,15). happensAt(walking(mws1gt_0),15). orientation(mws1gt_2,144,15). coords(mws1gt_2,47,116,15).  visible(mws1gt_2,15). happensAt(walking(mws1gt_2),15).
orientation(mws1gt_0,28,16). coords(mws1gt_0,279,263,16).  visible(mws1gt_0,16). happensAt(walking(mws1gt_0),16). orientation(mws1gt_2,144,16). coords(mws1gt_2,47,117,16).  visible(mws1gt_2,16). happensAt(walking(mws1gt_2),16).
orientation(mws1gt_0,39,17). coords(mws1gt_0,283,266,17).  visible(mws1gt_0,17). happensAt(walking(mws1gt_0),17). orientation(mws1gt_2,144,17). coords(mws1gt_2,47,118,17).  visible(mws1gt_2,17). happensAt(walking(mws1gt_2),17).
orientation(mws1gt_0,39,18). coords(mws1gt_0,286,266,18).  visible(mws1gt_0,18). happensAt(walking(mws1gt_0),18). orientation(mws1gt_2,155,18). coords(mws1gt_2,46,120,18).  visible(mws1gt_2,18). happensAt(walking(mws1gt_2),18).
orientation(mws1gt_0,39,19). coords(mws1gt_0,288,267,19).  visible(mws1gt_0,19). happensAt(walking(mws1gt_0),19). orientation(mws1gt_2,155,19). coords(mws1gt_2,46,120,19).  visible(mws1gt_2,19). happensAt(walking(mws1gt_2),19).
orientation(mws1gt_0,39,20). coords(mws1gt_0,291,267,20).  visible(mws1gt_0,20). happensAt(walking(mws1gt_0),20). orientation(mws1gt_2,155,20). coords(mws1gt_2,46,121,20).  visible(mws1gt_2,20). happensAt(walking(mws1gt_2),20).
orientation(mws1gt_0,39,21). coords(mws1gt_0,291,268,21).  visible(mws1gt_0,21). happensAt(walking(mws1gt_0),21). orientation(mws1gt_2,155,21). coords(mws1gt_2,46,122,21).  visible(mws1gt_2,21). happensAt(walking(mws1gt_2),21).
orientation(mws1gt_0,90,22). coords(mws1gt_0,292,269,22).  visible(mws1gt_0,22). happensAt(walking(mws1gt_0),22). orientation(mws1gt_2,155,22). coords(mws1gt_2,44,123,22).  visible(mws1gt_2,22). happensAt(walking(mws1gt_2),22).
orientation(mws1gt_0,90,23). coords(mws1gt_0,295,271,23).  visible(mws1gt_0,23). happensAt(walking(mws1gt_0),23). orientation(mws1gt_2,155,23). coords(mws1gt_2,44,124,23).  visible(mws1gt_2,23). happensAt(walking(mws1gt_2),23).
orientation(mws1gt_0,90,24). coords(mws1gt_0,298,273,24).  visible(mws1gt_0,24). happensAt(walking(mws1gt_0),24). orientation(mws1gt_2,155,24). coords(mws1gt_2,44,125,24).  visible(mws1gt_2,24). happensAt(walking(mws1gt_2),24).
orientation(mws1gt_0,90,25). coords(mws1gt_0,300,276,25).  visible(mws1gt_0,25). happensAt(walking(mws1gt_0),25). orientation(mws1gt_2,155,25). coords(mws1gt_2,44,126,25).  visible(mws1gt_2,25). happensAt(walking(mws1gt_2),25).
orientation(mws1gt_0,90,26). coords(mws1gt_0,300,278,26).  visible(mws1gt_0,26). happensAt(walking(mws1gt_0),26). orientation(mws1gt_2,155,26). coords(mws1gt_2,44,126,26).  visible(mws1gt_2,26). happensAt(walking(mws1gt_2),26).
orientation(mws1gt_0,90,27). coords(mws1gt_0,300,281,27).  happensAt(disappear(mws1gt_0),27). happensAt(walking(mws1gt_0),27). orientation(mws1gt_2,155,27). coords(mws1gt_2,44,127,27).  visible(mws1gt_2,27). happensAt(walking(mws1gt_2),27).
#show holdsAt/2.