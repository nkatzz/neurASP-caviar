
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

	orientation(rsfgt_1,170,0). coords(rsfgt_1,220,208,0).  happensAt(appear(rsfgt_1),0). happensAt(walking(rsfgt_1),0). orientation(rsfgt_2,170,0). coords(rsfgt_2,147,215,0).  happensAt(appear(rsfgt_2),0). happensAt(walking(rsfgt_2),0).
orientation(rsfgt_1,170,1). coords(rsfgt_1,220,207,1).  visible(rsfgt_1,1). happensAt(walking(rsfgt_1),1). orientation(rsfgt_2,170,1). coords(rsfgt_2,149,215,1).  visible(rsfgt_2,1). happensAt(walking(rsfgt_2),1).
orientation(rsfgt_1,170,2). coords(rsfgt_1,219,206,2).  visible(rsfgt_1,2). happensAt(walking(rsfgt_1),2). orientation(rsfgt_2,170,2). coords(rsfgt_2,149,215,2).  visible(rsfgt_2,2). happensAt(walking(rsfgt_2),2).
orientation(rsfgt_1,170,3). coords(rsfgt_1,219,205,3).  visible(rsfgt_1,3). happensAt(walking(rsfgt_1),3). orientation(rsfgt_2,170,3). coords(rsfgt_2,149,215,3).  visible(rsfgt_2,3). happensAt(walking(rsfgt_2),3).
orientation(rsfgt_1,170,4). coords(rsfgt_1,219,205,4).  visible(rsfgt_1,4). happensAt(walking(rsfgt_1),4). orientation(rsfgt_2,170,4). coords(rsfgt_2,151,215,4).  visible(rsfgt_2,4). happensAt(walking(rsfgt_2),4).
orientation(rsfgt_1,170,5). coords(rsfgt_1,218,205,5).  visible(rsfgt_1,5). happensAt(walking(rsfgt_1),5). orientation(rsfgt_2,170,5). coords(rsfgt_2,151,215,5).  visible(rsfgt_2,5). happensAt(walking(rsfgt_2),5).
orientation(rsfgt_1,170,6). coords(rsfgt_1,213,202,6).  visible(rsfgt_1,6). happensAt(walking(rsfgt_1),6). orientation(rsfgt_2,170,6). coords(rsfgt_2,153,215,6).  visible(rsfgt_2,6). happensAt(walking(rsfgt_2),6).
orientation(rsfgt_1,170,7). coords(rsfgt_1,213,202,7).  visible(rsfgt_1,7). happensAt(walking(rsfgt_1),7). orientation(rsfgt_2,170,7). coords(rsfgt_2,154,215,7).  visible(rsfgt_2,7). happensAt(walking(rsfgt_2),7).
orientation(rsfgt_1,170,8). coords(rsfgt_1,210,198,8).  visible(rsfgt_1,8). happensAt(walking(rsfgt_1),8). orientation(rsfgt_2,170,8). coords(rsfgt_2,154,215,8).  visible(rsfgt_2,8). happensAt(walking(rsfgt_2),8).
orientation(rsfgt_1,170,9). coords(rsfgt_1,209,194,9).  visible(rsfgt_1,9). happensAt(walking(rsfgt_1),9). orientation(rsfgt_2,170,9). coords(rsfgt_2,154,215,9).  visible(rsfgt_2,9). happensAt(walking(rsfgt_2),9).
orientation(rsfgt_1,170,10). coords(rsfgt_1,208,193,10).  visible(rsfgt_1,10). happensAt(walking(rsfgt_1),10). orientation(rsfgt_2,170,10). coords(rsfgt_2,157,215,10).  visible(rsfgt_2,10). happensAt(walking(rsfgt_2),10).
orientation(rsfgt_1,170,11). coords(rsfgt_1,207,188,11).  visible(rsfgt_1,11). happensAt(walking(rsfgt_1),11). orientation(rsfgt_2,170,11). coords(rsfgt_2,159,215,11).  visible(rsfgt_2,11). happensAt(walking(rsfgt_2),11).
orientation(rsfgt_1,170,12). coords(rsfgt_1,206,187,12).  visible(rsfgt_1,12). happensAt(walking(rsfgt_1),12). orientation(rsfgt_2,170,12). coords(rsfgt_2,162,215,12).  visible(rsfgt_2,12). happensAt(walking(rsfgt_2),12).
orientation(rsfgt_1,170,13). coords(rsfgt_1,206,187,13).  visible(rsfgt_1,13). happensAt(walking(rsfgt_1),13). orientation(rsfgt_2,170,13). coords(rsfgt_2,165,215,13).  visible(rsfgt_2,13). happensAt(walking(rsfgt_2),13).
orientation(rsfgt_1,170,14). coords(rsfgt_1,206,185,14).  visible(rsfgt_1,14). happensAt(walking(rsfgt_1),14). orientation(rsfgt_2,170,14). coords(rsfgt_2,167,215,14).  visible(rsfgt_2,14). happensAt(walking(rsfgt_2),14).
orientation(rsfgt_1,170,15). coords(rsfgt_1,205,183,15).  visible(rsfgt_1,15). happensAt(walking(rsfgt_1),15). orientation(rsfgt_2,170,15). coords(rsfgt_2,167,215,15).  visible(rsfgt_2,15). happensAt(walking(rsfgt_2),15).
orientation(rsfgt_1,170,16). coords(rsfgt_1,204,183,16).  visible(rsfgt_1,16). happensAt(walking(rsfgt_1),16). orientation(rsfgt_2,170,16). coords(rsfgt_2,168,215,16).  visible(rsfgt_2,16). happensAt(walking(rsfgt_2),16).
orientation(rsfgt_1,170,17). coords(rsfgt_1,204,181,17).  visible(rsfgt_1,17). happensAt(walking(rsfgt_1),17). orientation(rsfgt_2,170,17). coords(rsfgt_2,170,215,17).  visible(rsfgt_2,17). happensAt(walking(rsfgt_2),17).
orientation(rsfgt_1,170,18). coords(rsfgt_1,203,180,18).  visible(rsfgt_1,18). happensAt(walking(rsfgt_1),18). orientation(rsfgt_2,170,18). coords(rsfgt_2,171,215,18).  visible(rsfgt_2,18). happensAt(walking(rsfgt_2),18).
orientation(rsfgt_1,170,19). coords(rsfgt_1,202,179,19).  visible(rsfgt_1,19). happensAt(walking(rsfgt_1),19). orientation(rsfgt_2,170,19). coords(rsfgt_2,171,215,19).  visible(rsfgt_2,19). happensAt(walking(rsfgt_2),19).
orientation(rsfgt_1,170,20). coords(rsfgt_1,201,178,20).  visible(rsfgt_1,20). happensAt(walking(rsfgt_1),20). orientation(rsfgt_2,170,20). coords(rsfgt_2,172,215,20).  visible(rsfgt_2,20). happensAt(walking(rsfgt_2),20).
orientation(rsfgt_1,170,21). coords(rsfgt_1,198,175,21).  visible(rsfgt_1,21). happensAt(walking(rsfgt_1),21). orientation(rsfgt_2,170,21). coords(rsfgt_2,172,215,21).  visible(rsfgt_2,21). happensAt(walking(rsfgt_2),21).
orientation(rsfgt_1,170,22). coords(rsfgt_1,197,174,22).  visible(rsfgt_1,22). happensAt(walking(rsfgt_1),22). orientation(rsfgt_2,170,22). coords(rsfgt_2,173,215,22).  visible(rsfgt_2,22). happensAt(walking(rsfgt_2),22).
orientation(rsfgt_1,170,23). coords(rsfgt_1,196,172,23).  visible(rsfgt_1,23). happensAt(walking(rsfgt_1),23). orientation(rsfgt_2,170,23). coords(rsfgt_2,174,215,23).  visible(rsfgt_2,23). happensAt(walking(rsfgt_2),23).
orientation(rsfgt_1,170,24). coords(rsfgt_1,192,167,24).  visible(rsfgt_1,24). happensAt(walking(rsfgt_1),24). orientation(rsfgt_2,170,24). coords(rsfgt_2,180,217,24).  visible(rsfgt_2,24). happensAt(walking(rsfgt_2),24).
#show holdsAt/2.