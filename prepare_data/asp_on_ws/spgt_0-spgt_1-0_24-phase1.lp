
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

	orientation(spgt_0,160,0). coords(spgt_0,35,166,0).  happensAt(appear(spgt_0),0). happensAt(walking(spgt_0),0). orientation(spgt_1,160,0). coords(spgt_1,36,156,0).  happensAt(appear(spgt_1),0). happensAt(walking(spgt_1),0).
orientation(spgt_0,160,1). coords(spgt_0,35,166,1).  visible(spgt_0,1). happensAt(walking(spgt_0),1). orientation(spgt_1,160,1). coords(spgt_1,36,156,1).  visible(spgt_1,1). happensAt(walking(spgt_1),1).
orientation(spgt_0,160,2). coords(spgt_0,35,166,2).  visible(spgt_0,2). happensAt(walking(spgt_0),2). orientation(spgt_1,160,2). coords(spgt_1,36,156,2).  visible(spgt_1,2). happensAt(walking(spgt_1),2).
orientation(spgt_0,160,3). coords(spgt_0,35,166,3).  visible(spgt_0,3). happensAt(walking(spgt_0),3). orientation(spgt_1,160,3). coords(spgt_1,36,156,3).  visible(spgt_1,3). happensAt(walking(spgt_1),3).
orientation(spgt_0,160,4). coords(spgt_0,35,165,4).  visible(spgt_0,4). happensAt(walking(spgt_0),4). orientation(spgt_1,160,4). coords(spgt_1,36,156,4).  visible(spgt_1,4). happensAt(walking(spgt_1),4).
orientation(spgt_0,160,5). coords(spgt_0,35,165,5).  visible(spgt_0,5). happensAt(walking(spgt_0),5). orientation(spgt_1,160,5). coords(spgt_1,36,156,5).  visible(spgt_1,5). happensAt(walking(spgt_1),5).
orientation(spgt_0,160,6). coords(spgt_0,35,166,6).  visible(spgt_0,6). happensAt(walking(spgt_0),6). orientation(spgt_1,160,6). coords(spgt_1,37,156,6).  visible(spgt_1,6). happensAt(walking(spgt_1),6).
orientation(spgt_0,160,7). coords(spgt_0,35,166,7).  visible(spgt_0,7). happensAt(walking(spgt_0),7). orientation(spgt_1,160,7). coords(spgt_1,37,156,7).  visible(spgt_1,7). happensAt(walking(spgt_1),7).
orientation(spgt_0,160,8). coords(spgt_0,35,166,8).  visible(spgt_0,8). happensAt(walking(spgt_0),8). orientation(spgt_1,160,8). coords(spgt_1,37,156,8).  visible(spgt_1,8). happensAt(walking(spgt_1),8).
orientation(spgt_0,160,9). coords(spgt_0,36,166,9).  visible(spgt_0,9). happensAt(walking(spgt_0),9). orientation(spgt_1,160,9). coords(spgt_1,38,156,9).  visible(spgt_1,9). happensAt(walking(spgt_1),9).
orientation(spgt_0,160,10). coords(spgt_0,36,166,10).  visible(spgt_0,10). happensAt(walking(spgt_0),10). orientation(spgt_1,160,10). coords(spgt_1,38,156,10).  visible(spgt_1,10). happensAt(walking(spgt_1),10).
orientation(spgt_0,160,11). coords(spgt_0,37,166,11).  visible(spgt_0,11). happensAt(walking(spgt_0),11). orientation(spgt_1,160,11). coords(spgt_1,38,157,11).  visible(spgt_1,11). happensAt(walking(spgt_1),11).
orientation(spgt_0,160,12). coords(spgt_0,37,166,12).  visible(spgt_0,12). happensAt(walking(spgt_0),12). orientation(spgt_1,160,12). coords(spgt_1,38,157,12).  visible(spgt_1,12). happensAt(walking(spgt_1),12).
orientation(spgt_0,160,13). coords(spgt_0,37,166,13).  visible(spgt_0,13). happensAt(walking(spgt_0),13). orientation(spgt_1,160,13). coords(spgt_1,38,157,13).  visible(spgt_1,13). happensAt(walking(spgt_1),13).
orientation(spgt_0,156,14). coords(spgt_0,37,166,14).  visible(spgt_0,14). happensAt(walking(spgt_0),14). orientation(spgt_1,160,14). coords(spgt_1,38,157,14).  visible(spgt_1,14). happensAt(walking(spgt_1),14).
orientation(spgt_0,156,15). coords(spgt_0,37,166,15).  visible(spgt_0,15). happensAt(walking(spgt_0),15). orientation(spgt_1,160,15). coords(spgt_1,39,157,15).  visible(spgt_1,15). happensAt(walking(spgt_1),15).
orientation(spgt_0,156,16). coords(spgt_0,37,167,16).  visible(spgt_0,16). happensAt(walking(spgt_0),16). orientation(spgt_1,160,16). coords(spgt_1,39,158,16).  visible(spgt_1,16). happensAt(walking(spgt_1),16).
orientation(spgt_0,156,17). coords(spgt_0,38,167,17).  visible(spgt_0,17). happensAt(walking(spgt_0),17). orientation(spgt_1,160,17). coords(spgt_1,40,158,17).  visible(spgt_1,17). happensAt(walking(spgt_1),17).
orientation(spgt_0,156,18). coords(spgt_0,38,167,18).  visible(spgt_0,18). happensAt(walking(spgt_0),18). orientation(spgt_1,160,18). coords(spgt_1,40,158,18).  visible(spgt_1,18). happensAt(walking(spgt_1),18).
orientation(spgt_0,156,19). coords(spgt_0,38,167,19).  visible(spgt_0,19). happensAt(walking(spgt_0),19). orientation(spgt_1,160,19). coords(spgt_1,40,158,19).  visible(spgt_1,19). happensAt(walking(spgt_1),19).
orientation(spgt_0,156,20). coords(spgt_0,38,167,20).  visible(spgt_0,20). happensAt(walking(spgt_0),20). orientation(spgt_1,160,20). coords(spgt_1,40,158,20).  visible(spgt_1,20). happensAt(walking(spgt_1),20).
orientation(spgt_0,156,21). coords(spgt_0,39,167,21).  visible(spgt_0,21). happensAt(walking(spgt_0),21). orientation(spgt_1,160,21). coords(spgt_1,40,158,21).  visible(spgt_1,21). happensAt(walking(spgt_1),21).
orientation(spgt_0,156,22). coords(spgt_0,39,167,22).  visible(spgt_0,22). happensAt(walking(spgt_0),22). orientation(spgt_1,160,22). coords(spgt_1,40,158,22).  visible(spgt_1,22). happensAt(walking(spgt_1),22).
orientation(spgt_0,156,23). coords(spgt_0,41,167,23).  visible(spgt_0,23). happensAt(walking(spgt_0),23). orientation(spgt_1,160,23). coords(spgt_1,40,158,23).  visible(spgt_1,23). happensAt(walking(spgt_1),23).
orientation(spgt_0,156,24). coords(spgt_0,41,167,24).  visible(spgt_0,24). happensAt(walking(spgt_0),24). orientation(spgt_1,160,24). coords(spgt_1,41,158,24).  visible(spgt_1,24). happensAt(walking(spgt_1),24).
#show holdsAt/2.