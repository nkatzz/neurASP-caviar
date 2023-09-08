
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

	orientation(mws1gt_0,170,3). coords(mws1gt_0,253,217,3).  visible(mws1gt_0,3). happensAt(walking(mws1gt_0),3). orientation(mws1gt_1,170,3). coords(mws1gt_1,65,193,3).  visible(mws1gt_1,3). happensAt(inactive(mws1gt_1),3).
orientation(mws1gt_0,170,4). coords(mws1gt_0,257,219,4).  visible(mws1gt_0,4). happensAt(walking(mws1gt_0),4). orientation(mws1gt_1,170,4). coords(mws1gt_1,65,193,4).  visible(mws1gt_1,4). happensAt(inactive(mws1gt_1),4).
orientation(mws1gt_0,175,5). coords(mws1gt_0,260,226,5).  visible(mws1gt_0,5). happensAt(walking(mws1gt_0),5). orientation(mws1gt_1,170,5). coords(mws1gt_1,65,193,5).  visible(mws1gt_1,5). happensAt(inactive(mws1gt_1),5).
orientation(mws1gt_0,175,6). coords(mws1gt_0,260,229,6).  visible(mws1gt_0,6). happensAt(walking(mws1gt_0),6). orientation(mws1gt_1,170,6). coords(mws1gt_1,65,193,6).  visible(mws1gt_1,6). happensAt(inactive(mws1gt_1),6).
orientation(mws1gt_0,175,7). coords(mws1gt_0,261,230,7).  visible(mws1gt_0,7). happensAt(walking(mws1gt_0),7). orientation(mws1gt_1,170,7). coords(mws1gt_1,65,193,7).  visible(mws1gt_1,7). happensAt(inactive(mws1gt_1),7).
orientation(mws1gt_0,175,8). coords(mws1gt_0,263,233,8).  visible(mws1gt_0,8). happensAt(walking(mws1gt_0),8). orientation(mws1gt_1,170,8). coords(mws1gt_1,65,193,8).  visible(mws1gt_1,8). happensAt(inactive(mws1gt_1),8).
orientation(mws1gt_0,19,9). coords(mws1gt_0,265,237,9).  visible(mws1gt_0,9). happensAt(walking(mws1gt_0),9). orientation(mws1gt_1,170,9). coords(mws1gt_1,65,193,9).  visible(mws1gt_1,9). happensAt(inactive(mws1gt_1),9).
orientation(mws1gt_0,28,10). coords(mws1gt_0,266,239,10).  visible(mws1gt_0,10). happensAt(walking(mws1gt_0),10). orientation(mws1gt_1,170,10). coords(mws1gt_1,65,193,10).  visible(mws1gt_1,10). happensAt(inactive(mws1gt_1),10).
orientation(mws1gt_0,28,11). coords(mws1gt_0,268,241,11).  visible(mws1gt_0,11). happensAt(walking(mws1gt_0),11). orientation(mws1gt_1,170,11). coords(mws1gt_1,65,193,11).  visible(mws1gt_1,11). happensAt(inactive(mws1gt_1),11).
orientation(mws1gt_0,28,12). coords(mws1gt_0,269,244,12).  visible(mws1gt_0,12). happensAt(walking(mws1gt_0),12). orientation(mws1gt_1,170,12). coords(mws1gt_1,65,193,12).  visible(mws1gt_1,12). happensAt(inactive(mws1gt_1),12).
orientation(mws1gt_0,28,13). coords(mws1gt_0,271,248,13).  visible(mws1gt_0,13). happensAt(walking(mws1gt_0),13). orientation(mws1gt_1,170,13). coords(mws1gt_1,65,193,13).  visible(mws1gt_1,13). happensAt(inactive(mws1gt_1),13).
orientation(mws1gt_0,28,14). coords(mws1gt_0,274,253,14).  visible(mws1gt_0,14). happensAt(walking(mws1gt_0),14). orientation(mws1gt_1,170,14). coords(mws1gt_1,65,193,14).  visible(mws1gt_1,14). happensAt(inactive(mws1gt_1),14).
orientation(mws1gt_0,28,15). coords(mws1gt_0,275,258,15).  visible(mws1gt_0,15). happensAt(walking(mws1gt_0),15). orientation(mws1gt_1,170,15). coords(mws1gt_1,65,193,15).  visible(mws1gt_1,15). happensAt(inactive(mws1gt_1),15).
orientation(mws1gt_0,28,16). coords(mws1gt_0,279,263,16).  visible(mws1gt_0,16). happensAt(walking(mws1gt_0),16). orientation(mws1gt_1,170,16). coords(mws1gt_1,65,193,16).  visible(mws1gt_1,16). happensAt(inactive(mws1gt_1),16).
orientation(mws1gt_0,39,17). coords(mws1gt_0,283,266,17).  visible(mws1gt_0,17). happensAt(walking(mws1gt_0),17). orientation(mws1gt_1,170,17). coords(mws1gt_1,65,193,17).  visible(mws1gt_1,17). happensAt(inactive(mws1gt_1),17).
orientation(mws1gt_0,39,18). coords(mws1gt_0,286,266,18).  visible(mws1gt_0,18). happensAt(walking(mws1gt_0),18). orientation(mws1gt_1,170,18). coords(mws1gt_1,65,193,18).  visible(mws1gt_1,18). happensAt(inactive(mws1gt_1),18).
orientation(mws1gt_0,39,19). coords(mws1gt_0,288,267,19).  visible(mws1gt_0,19). happensAt(walking(mws1gt_0),19). orientation(mws1gt_1,170,19). coords(mws1gt_1,65,193,19).  visible(mws1gt_1,19). happensAt(inactive(mws1gt_1),19).
orientation(mws1gt_0,39,20). coords(mws1gt_0,291,267,20).  visible(mws1gt_0,20). happensAt(walking(mws1gt_0),20). orientation(mws1gt_1,170,20). coords(mws1gt_1,65,193,20).  visible(mws1gt_1,20). happensAt(inactive(mws1gt_1),20).
orientation(mws1gt_0,39,21). coords(mws1gt_0,291,268,21).  visible(mws1gt_0,21). happensAt(walking(mws1gt_0),21). orientation(mws1gt_1,170,21). coords(mws1gt_1,65,193,21).  visible(mws1gt_1,21). happensAt(inactive(mws1gt_1),21).
orientation(mws1gt_0,90,22). coords(mws1gt_0,292,269,22).  visible(mws1gt_0,22). happensAt(walking(mws1gt_0),22). orientation(mws1gt_1,170,22). coords(mws1gt_1,65,193,22).  visible(mws1gt_1,22). happensAt(inactive(mws1gt_1),22).
orientation(mws1gt_0,90,23). coords(mws1gt_0,295,271,23).  visible(mws1gt_0,23). happensAt(walking(mws1gt_0),23). orientation(mws1gt_1,170,23). coords(mws1gt_1,65,193,23).  visible(mws1gt_1,23). happensAt(inactive(mws1gt_1),23).
orientation(mws1gt_0,90,24). coords(mws1gt_0,298,273,24).  visible(mws1gt_0,24). happensAt(walking(mws1gt_0),24). orientation(mws1gt_1,170,24). coords(mws1gt_1,65,193,24).  visible(mws1gt_1,24). happensAt(inactive(mws1gt_1),24).
orientation(mws1gt_0,90,25). coords(mws1gt_0,300,276,25).  visible(mws1gt_0,25). happensAt(walking(mws1gt_0),25). orientation(mws1gt_1,170,25). coords(mws1gt_1,65,193,25).  visible(mws1gt_1,25). happensAt(inactive(mws1gt_1),25).
orientation(mws1gt_0,90,26). coords(mws1gt_0,300,278,26).  visible(mws1gt_0,26). happensAt(walking(mws1gt_0),26). orientation(mws1gt_1,170,26). coords(mws1gt_1,65,193,26).  visible(mws1gt_1,26). happensAt(inactive(mws1gt_1),26).
orientation(mws1gt_0,90,27). coords(mws1gt_0,300,281,27).  happensAt(disappear(mws1gt_0),27). happensAt(walking(mws1gt_0),27). orientation(mws1gt_1,170,27). coords(mws1gt_1,65,193,27).  visible(mws1gt_1,27). happensAt(inactive(mws1gt_1),27).
#show holdsAt/2.