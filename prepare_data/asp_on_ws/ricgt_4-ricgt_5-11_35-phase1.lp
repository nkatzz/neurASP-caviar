
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

	orientation(ricgt_4,172,11). coords(ricgt_4,33,116,11).  visible(ricgt_4,11). happensAt(walking(ricgt_4),11). orientation(ricgt_5,0,11). coords(ricgt_5,30,112,11).  happensAt(appear(ricgt_5),11). happensAt(walking(ricgt_5),11).
orientation(ricgt_4,172,12). coords(ricgt_4,33,116,12).  visible(ricgt_4,12). happensAt(walking(ricgt_4),12). orientation(ricgt_5,0,12). coords(ricgt_5,30,112,12).  visible(ricgt_5,12). happensAt(walking(ricgt_5),12).
orientation(ricgt_4,172,13). coords(ricgt_4,33,117,13).  visible(ricgt_4,13). happensAt(walking(ricgt_4),13). orientation(ricgt_5,0,13). coords(ricgt_5,30,112,13).  visible(ricgt_5,13). happensAt(walking(ricgt_5),13).
orientation(ricgt_4,172,14). coords(ricgt_4,33,117,14).  visible(ricgt_4,14). happensAt(walking(ricgt_4),14). orientation(ricgt_5,0,14). coords(ricgt_5,30,112,14).  visible(ricgt_5,14). happensAt(walking(ricgt_5),14).
orientation(ricgt_4,172,15). coords(ricgt_4,33,117,15).  visible(ricgt_4,15). happensAt(walking(ricgt_4),15). orientation(ricgt_5,0,15). coords(ricgt_5,30,112,15).  visible(ricgt_5,15). happensAt(walking(ricgt_5),15).
orientation(ricgt_4,172,16). coords(ricgt_4,32,117,16).  visible(ricgt_4,16). happensAt(walking(ricgt_4),16). orientation(ricgt_5,0,16). coords(ricgt_5,30,112,16).  visible(ricgt_5,16). happensAt(walking(ricgt_5),16).
orientation(ricgt_4,172,17). coords(ricgt_4,32,117,17).  visible(ricgt_4,17). happensAt(walking(ricgt_4),17). orientation(ricgt_5,0,17). coords(ricgt_5,30,112,17).  visible(ricgt_5,17). happensAt(walking(ricgt_5),17).
orientation(ricgt_4,172,18). coords(ricgt_4,32,117,18).  visible(ricgt_4,18). happensAt(walking(ricgt_4),18). orientation(ricgt_5,0,18). coords(ricgt_5,31,112,18).  visible(ricgt_5,18). happensAt(walking(ricgt_5),18).
orientation(ricgt_4,172,19). coords(ricgt_4,32,117,19).  visible(ricgt_4,19). happensAt(walking(ricgt_4),19). orientation(ricgt_5,0,19). coords(ricgt_5,31,113,19).  visible(ricgt_5,19). happensAt(walking(ricgt_5),19).
orientation(ricgt_4,172,20). coords(ricgt_4,32,117,20).  visible(ricgt_4,20). happensAt(walking(ricgt_4),20). orientation(ricgt_5,0,20). coords(ricgt_5,31,113,20).  visible(ricgt_5,20). happensAt(walking(ricgt_5),20).
orientation(ricgt_4,172,21). coords(ricgt_4,32,118,21).  visible(ricgt_4,21). happensAt(walking(ricgt_4),21). orientation(ricgt_5,0,21). coords(ricgt_5,30,113,21).  visible(ricgt_5,21). happensAt(walking(ricgt_5),21).
orientation(ricgt_4,172,22). coords(ricgt_4,32,118,22).  visible(ricgt_4,22). happensAt(walking(ricgt_4),22). orientation(ricgt_5,0,22). coords(ricgt_5,30,114,22).  visible(ricgt_5,22). happensAt(walking(ricgt_5),22).
orientation(ricgt_4,172,23). coords(ricgt_4,32,118,23).  visible(ricgt_4,23). happensAt(walking(ricgt_4),23). orientation(ricgt_5,0,23). coords(ricgt_5,30,114,23).  visible(ricgt_5,23). happensAt(walking(ricgt_5),23).
orientation(ricgt_4,172,24). coords(ricgt_4,31,118,24).  visible(ricgt_4,24). happensAt(walking(ricgt_4),24). orientation(ricgt_5,0,24). coords(ricgt_5,30,114,24).  visible(ricgt_5,24). happensAt(walking(ricgt_5),24).
orientation(ricgt_4,172,25). coords(ricgt_4,31,118,25).  visible(ricgt_4,25). happensAt(walking(ricgt_4),25). orientation(ricgt_5,0,25). coords(ricgt_5,30,114,25).  visible(ricgt_5,25). happensAt(walking(ricgt_5),25).
orientation(ricgt_4,172,26). coords(ricgt_4,31,118,26).  visible(ricgt_4,26). happensAt(walking(ricgt_4),26). orientation(ricgt_5,0,26). coords(ricgt_5,30,114,26).  visible(ricgt_5,26). happensAt(walking(ricgt_5),26).
orientation(ricgt_4,172,27). coords(ricgt_4,31,118,27).  visible(ricgt_4,27). happensAt(walking(ricgt_4),27). orientation(ricgt_5,0,27). coords(ricgt_5,30,114,27).  visible(ricgt_5,27). happensAt(walking(ricgt_5),27).
orientation(ricgt_4,172,28). coords(ricgt_4,31,118,28).  visible(ricgt_4,28). happensAt(walking(ricgt_4),28). orientation(ricgt_5,0,28). coords(ricgt_5,30,114,28).  visible(ricgt_5,28). happensAt(walking(ricgt_5),28).
orientation(ricgt_4,172,29). coords(ricgt_4,31,118,29).  visible(ricgt_4,29). happensAt(walking(ricgt_4),29). orientation(ricgt_5,0,29). coords(ricgt_5,30,114,29).  visible(ricgt_5,29). happensAt(walking(ricgt_5),29).
orientation(ricgt_4,172,30). coords(ricgt_4,31,118,30).  visible(ricgt_4,30). happensAt(walking(ricgt_4),30). orientation(ricgt_5,0,30). coords(ricgt_5,30,114,30).  visible(ricgt_5,30). happensAt(walking(ricgt_5),30).
orientation(ricgt_4,172,31). coords(ricgt_4,31,120,31).  visible(ricgt_4,31). happensAt(walking(ricgt_4),31). orientation(ricgt_5,0,31). coords(ricgt_5,30,114,31).  visible(ricgt_5,31). happensAt(walking(ricgt_5),31).
orientation(ricgt_4,172,32). coords(ricgt_4,31,120,32).  visible(ricgt_4,32). happensAt(walking(ricgt_4),32). orientation(ricgt_5,0,32). coords(ricgt_5,30,115,32).  visible(ricgt_5,32). happensAt(walking(ricgt_5),32).
orientation(ricgt_4,172,33). coords(ricgt_4,31,121,33).  visible(ricgt_4,33). happensAt(walking(ricgt_4),33). orientation(ricgt_5,0,33). coords(ricgt_5,30,116,33).  visible(ricgt_5,33). happensAt(walking(ricgt_5),33).
orientation(ricgt_4,172,34). coords(ricgt_4,31,121,34).  visible(ricgt_4,34). happensAt(walking(ricgt_4),34). orientation(ricgt_5,0,34). coords(ricgt_5,30,116,34).  visible(ricgt_5,34). happensAt(walking(ricgt_5),34).
orientation(ricgt_4,172,35). coords(ricgt_4,31,122,35).  visible(ricgt_4,35). happensAt(walking(ricgt_4),35). orientation(ricgt_5,0,35). coords(ricgt_5,30,117,35).  visible(ricgt_5,35). happensAt(walking(ricgt_5),35).
#show holdsAt/2.