
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

	orientation(rffgt_0,142,0). coords(rffgt_0,150,71,0).  happensAt(appear(rffgt_0),0). happensAt(active(rffgt_0),0). orientation(rffgt_1,145,0). coords(rffgt_1,178,110,0).  happensAt(appear(rffgt_1),0). happensAt(active(rffgt_1),0).
orientation(rffgt_0,142,1). coords(rffgt_0,150,71,1).  visible(rffgt_0,1). happensAt(active(rffgt_0),1). orientation(rffgt_1,145,1). coords(rffgt_1,178,110,1).  visible(rffgt_1,1). happensAt(active(rffgt_1),1).
orientation(rffgt_0,142,2). coords(rffgt_0,151,71,2).  visible(rffgt_0,2). happensAt(active(rffgt_0),2). orientation(rffgt_1,145,2). coords(rffgt_1,176,110,2).  visible(rffgt_1,2). happensAt(active(rffgt_1),2).
orientation(rffgt_0,142,3). coords(rffgt_0,150,71,3).  visible(rffgt_0,3). happensAt(active(rffgt_0),3). orientation(rffgt_1,145,3). coords(rffgt_1,175,110,3).  visible(rffgt_1,3). happensAt(active(rffgt_1),3).
orientation(rffgt_0,143,4). coords(rffgt_0,150,71,4).  visible(rffgt_0,4). happensAt(active(rffgt_0),4). orientation(rffgt_1,145,4). coords(rffgt_1,175,110,4).  visible(rffgt_1,4). happensAt(active(rffgt_1),4).
orientation(rffgt_0,143,5). coords(rffgt_0,149,70,5).  visible(rffgt_0,5). happensAt(active(rffgt_0),5). orientation(rffgt_1,145,5). coords(rffgt_1,175,110,5).  visible(rffgt_1,5). happensAt(active(rffgt_1),5).
orientation(rffgt_0,143,6). coords(rffgt_0,148,71,6).  visible(rffgt_0,6). happensAt(active(rffgt_0),6). orientation(rffgt_1,148,6). coords(rffgt_1,174,110,6).  visible(rffgt_1,6). happensAt(active(rffgt_1),6).
orientation(rffgt_0,144,7). coords(rffgt_0,147,71,7).  visible(rffgt_0,7). happensAt(active(rffgt_0),7). orientation(rffgt_1,148,7). coords(rffgt_1,173,110,7).  visible(rffgt_1,7). happensAt(active(rffgt_1),7).
orientation(rffgt_0,140,8). coords(rffgt_0,147,71,8).  visible(rffgt_0,8). happensAt(active(rffgt_0),8). orientation(rffgt_1,148,8). coords(rffgt_1,172,110,8).  visible(rffgt_1,8). happensAt(active(rffgt_1),8).
orientation(rffgt_0,140,9). coords(rffgt_0,145,71,9).  visible(rffgt_0,9). happensAt(active(rffgt_0),9). orientation(rffgt_1,148,9). coords(rffgt_1,171,110,9).  visible(rffgt_1,9). happensAt(active(rffgt_1),9).
orientation(rffgt_0,140,10). coords(rffgt_0,145,71,10).  visible(rffgt_0,10). happensAt(active(rffgt_0),10). orientation(rffgt_1,148,10). coords(rffgt_1,169,110,10).  visible(rffgt_1,10). happensAt(active(rffgt_1),10).
orientation(rffgt_0,140,11). coords(rffgt_0,145,71,11).  visible(rffgt_0,11). happensAt(active(rffgt_0),11). orientation(rffgt_1,147,11). coords(rffgt_1,169,110,11).  visible(rffgt_1,11). happensAt(active(rffgt_1),11).
orientation(rffgt_0,140,12). coords(rffgt_0,145,71,12).  visible(rffgt_0,12). happensAt(active(rffgt_0),12). orientation(rffgt_1,146,12). coords(rffgt_1,168,109,12).  visible(rffgt_1,12). happensAt(active(rffgt_1),12).
orientation(rffgt_0,140,13). coords(rffgt_0,145,71,13).  visible(rffgt_0,13). happensAt(active(rffgt_0),13). orientation(rffgt_1,146,13). coords(rffgt_1,167,109,13).  visible(rffgt_1,13). happensAt(active(rffgt_1),13).
orientation(rffgt_0,140,14). coords(rffgt_0,144,71,14).  visible(rffgt_0,14). happensAt(active(rffgt_0),14). orientation(rffgt_1,144,14). coords(rffgt_1,167,109,14).  visible(rffgt_1,14). happensAt(active(rffgt_1),14).
orientation(rffgt_0,140,15). coords(rffgt_0,144,70,15).  visible(rffgt_0,15). happensAt(active(rffgt_0),15). orientation(rffgt_1,144,15). coords(rffgt_1,167,109,15).  visible(rffgt_1,15). happensAt(active(rffgt_1),15).
orientation(rffgt_0,140,16). coords(rffgt_0,144,70,16).  visible(rffgt_0,16). happensAt(active(rffgt_0),16). orientation(rffgt_1,144,16). coords(rffgt_1,167,108,16).  visible(rffgt_1,16). happensAt(active(rffgt_1),16).
orientation(rffgt_0,140,17). coords(rffgt_0,144,70,17).  visible(rffgt_0,17). happensAt(active(rffgt_0),17). orientation(rffgt_1,144,17). coords(rffgt_1,167,108,17).  visible(rffgt_1,17). happensAt(active(rffgt_1),17).
orientation(rffgt_0,140,18). coords(rffgt_0,144,70,18).  visible(rffgt_0,18). happensAt(active(rffgt_0),18). orientation(rffgt_1,144,18). coords(rffgt_1,167,108,18).  visible(rffgt_1,18). happensAt(active(rffgt_1),18).
orientation(rffgt_0,140,19). coords(rffgt_0,144,70,19).  visible(rffgt_0,19). happensAt(active(rffgt_0),19). orientation(rffgt_1,141,19). coords(rffgt_1,167,107,19).  visible(rffgt_1,19). happensAt(active(rffgt_1),19).
orientation(rffgt_0,140,20). coords(rffgt_0,143,70,20).  visible(rffgt_0,20). happensAt(active(rffgt_0),20). orientation(rffgt_1,141,20). coords(rffgt_1,167,107,20).  visible(rffgt_1,20). happensAt(active(rffgt_1),20).
orientation(rffgt_0,140,21). coords(rffgt_0,143,70,21).  visible(rffgt_0,21). happensAt(active(rffgt_0),21). orientation(rffgt_1,141,21). coords(rffgt_1,166,106,21).  visible(rffgt_1,21). happensAt(active(rffgt_1),21).
orientation(rffgt_0,140,22). coords(rffgt_0,143,70,22).  visible(rffgt_0,22). happensAt(active(rffgt_0),22). orientation(rffgt_1,141,22). coords(rffgt_1,166,106,22).  visible(rffgt_1,22). happensAt(active(rffgt_1),22).
orientation(rffgt_0,140,23). coords(rffgt_0,143,70,23).  visible(rffgt_0,23). happensAt(active(rffgt_0),23). orientation(rffgt_1,141,23). coords(rffgt_1,166,106,23).  visible(rffgt_1,23). happensAt(active(rffgt_1),23).
orientation(rffgt_0,144,24). coords(rffgt_0,143,69,24).  visible(rffgt_0,24). happensAt(active(rffgt_0),24). orientation(rffgt_1,141,24). coords(rffgt_1,166,105,24).  visible(rffgt_1,24). happensAt(active(rffgt_1),24).
#show holdsAt/2.