
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

	orientation(ricgt_0,51,0). coords(ricgt_0,298,257,0).  happensAt(appear(ricgt_0),0). happensAt(walking(ricgt_0),0). orientation(ricgt_4,161,0). coords(ricgt_4,39,114,0).  happensAt(appear(ricgt_4),0). happensAt(walking(ricgt_4),0).
orientation(ricgt_0,51,1). coords(ricgt_0,297,259,1).  visible(ricgt_0,1). happensAt(walking(ricgt_0),1). orientation(ricgt_4,161,1). coords(ricgt_4,38,114,1).  visible(ricgt_4,1). happensAt(walking(ricgt_4),1).
orientation(ricgt_0,49,2). coords(ricgt_0,296,262,2).  visible(ricgt_0,2). happensAt(walking(ricgt_0),2). orientation(ricgt_4,161,2). coords(ricgt_4,35,113,2).  visible(ricgt_4,2). happensAt(walking(ricgt_4),2).
orientation(ricgt_0,46,3). coords(ricgt_0,296,265,3).  visible(ricgt_0,3). happensAt(walking(ricgt_0),3). orientation(ricgt_4,172,3). coords(ricgt_4,35,114,3).  visible(ricgt_4,3). happensAt(walking(ricgt_4),3).
orientation(ricgt_0,43,4). coords(ricgt_0,294,268,4).  visible(ricgt_0,4). happensAt(walking(ricgt_0),4). orientation(ricgt_4,172,4). coords(ricgt_4,35,115,4).  visible(ricgt_4,4). happensAt(walking(ricgt_4),4).
orientation(ricgt_0,43,5). coords(ricgt_0,293,270,5).  visible(ricgt_0,5). happensAt(walking(ricgt_0),5). orientation(ricgt_4,172,5). coords(ricgt_4,34,115,5).  visible(ricgt_4,5). happensAt(walking(ricgt_4),5).
orientation(ricgt_0,43,6). coords(ricgt_0,291,272,6).  visible(ricgt_0,6). happensAt(walking(ricgt_0),6). orientation(ricgt_4,172,6). coords(ricgt_4,33,115,6).  visible(ricgt_4,6). happensAt(walking(ricgt_4),6).
orientation(ricgt_0,42,7). coords(ricgt_0,289,274,7).  visible(ricgt_0,7). happensAt(walking(ricgt_0),7). orientation(ricgt_4,172,7). coords(ricgt_4,33,116,7).  visible(ricgt_4,7). happensAt(walking(ricgt_4),7).
orientation(ricgt_0,42,8). coords(ricgt_0,289,274,8).  visible(ricgt_0,8). happensAt(walking(ricgt_0),8). orientation(ricgt_4,172,8). coords(ricgt_4,33,116,8).  visible(ricgt_4,8). happensAt(walking(ricgt_4),8).
orientation(ricgt_0,43,9). coords(ricgt_0,289,273,9).  visible(ricgt_0,9). happensAt(walking(ricgt_0),9). orientation(ricgt_4,172,9). coords(ricgt_4,33,116,9).  visible(ricgt_4,9). happensAt(walking(ricgt_4),9).
orientation(ricgt_0,43,10). coords(ricgt_0,287,274,10).  visible(ricgt_0,10). happensAt(walking(ricgt_0),10). orientation(ricgt_4,172,10). coords(ricgt_4,33,116,10).  visible(ricgt_4,10). happensAt(walking(ricgt_4),10).
orientation(ricgt_0,43,11). coords(ricgt_0,287,273,11).  visible(ricgt_0,11). happensAt(walking(ricgt_0),11). orientation(ricgt_4,172,11). coords(ricgt_4,33,116,11).  visible(ricgt_4,11). happensAt(walking(ricgt_4),11).
orientation(ricgt_0,43,12). coords(ricgt_0,286,273,12).  visible(ricgt_0,12). happensAt(walking(ricgt_0),12). orientation(ricgt_4,172,12). coords(ricgt_4,33,116,12).  visible(ricgt_4,12). happensAt(walking(ricgt_4),12).
orientation(ricgt_0,43,13). coords(ricgt_0,286,273,13).  visible(ricgt_0,13). happensAt(walking(ricgt_0),13). orientation(ricgt_4,172,13). coords(ricgt_4,33,117,13).  visible(ricgt_4,13). happensAt(walking(ricgt_4),13).
orientation(ricgt_0,43,14). coords(ricgt_0,285,274,14).  visible(ricgt_0,14). happensAt(walking(ricgt_0),14). orientation(ricgt_4,172,14). coords(ricgt_4,33,117,14).  visible(ricgt_4,14). happensAt(walking(ricgt_4),14).
orientation(ricgt_0,43,15). coords(ricgt_0,284,274,15).  visible(ricgt_0,15). happensAt(walking(ricgt_0),15). orientation(ricgt_4,172,15). coords(ricgt_4,33,117,15).  visible(ricgt_4,15). happensAt(walking(ricgt_4),15).
orientation(ricgt_0,43,16). coords(ricgt_0,283,275,16).  visible(ricgt_0,16). happensAt(walking(ricgt_0),16). orientation(ricgt_4,172,16). coords(ricgt_4,32,117,16).  visible(ricgt_4,16). happensAt(walking(ricgt_4),16).
orientation(ricgt_0,37,17). coords(ricgt_0,279,276,17).  visible(ricgt_0,17). happensAt(walking(ricgt_0),17). orientation(ricgt_4,172,17). coords(ricgt_4,32,117,17).  visible(ricgt_4,17). happensAt(walking(ricgt_4),17).
orientation(ricgt_0,37,18). coords(ricgt_0,278,277,18).  visible(ricgt_0,18). happensAt(walking(ricgt_0),18). orientation(ricgt_4,172,18). coords(ricgt_4,32,117,18).  visible(ricgt_4,18). happensAt(walking(ricgt_4),18).
orientation(ricgt_0,37,19). coords(ricgt_0,275,279,19).  visible(ricgt_0,19). happensAt(walking(ricgt_0),19). orientation(ricgt_4,172,19). coords(ricgt_4,32,117,19).  visible(ricgt_4,19). happensAt(walking(ricgt_4),19).
orientation(ricgt_0,37,20). coords(ricgt_0,274,280,20).  visible(ricgt_0,20). happensAt(walking(ricgt_0),20). orientation(ricgt_4,172,20). coords(ricgt_4,32,117,20).  visible(ricgt_4,20). happensAt(walking(ricgt_4),20).
orientation(ricgt_0,37,21). coords(ricgt_0,276,281,21).  visible(ricgt_0,21). happensAt(walking(ricgt_0),21). orientation(ricgt_4,172,21). coords(ricgt_4,32,118,21).  visible(ricgt_4,21). happensAt(walking(ricgt_4),21).
orientation(ricgt_0,37,22). coords(ricgt_0,279,283,22).  visible(ricgt_0,22). happensAt(walking(ricgt_0),22). orientation(ricgt_4,172,22). coords(ricgt_4,32,118,22).  visible(ricgt_4,22). happensAt(walking(ricgt_4),22).
orientation(ricgt_0,37,23). coords(ricgt_0,278,284,23).  visible(ricgt_0,23). happensAt(walking(ricgt_0),23). orientation(ricgt_4,172,23). coords(ricgt_4,32,118,23).  visible(ricgt_4,23). happensAt(walking(ricgt_4),23).
orientation(ricgt_0,37,24). coords(ricgt_0,279,284,24).  visible(ricgt_0,24). happensAt(walking(ricgt_0),24). orientation(ricgt_4,172,24). coords(ricgt_4,31,118,24).  visible(ricgt_4,24). happensAt(walking(ricgt_4),24).
#show holdsAt/2.