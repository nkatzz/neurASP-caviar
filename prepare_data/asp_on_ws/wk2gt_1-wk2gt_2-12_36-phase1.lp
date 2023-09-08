
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

	orientation(wk2gt_1,143,12). coords(wk2gt_1,103,65,12).  visible(wk2gt_1,12). happensAt(walking(wk2gt_1),12). orientation(wk2gt_2,141,12). coords(wk2gt_2,95,76,12).  visible(wk2gt_2,12). happensAt(walking(wk2gt_2),12).
orientation(wk2gt_1,143,13). coords(wk2gt_1,103,65,13).  visible(wk2gt_1,13). happensAt(walking(wk2gt_1),13). orientation(wk2gt_2,141,13). coords(wk2gt_2,95,75,13).  visible(wk2gt_2,13). happensAt(walking(wk2gt_2),13).
orientation(wk2gt_1,143,14). coords(wk2gt_1,104,65,14).  visible(wk2gt_1,14). happensAt(walking(wk2gt_1),14). orientation(wk2gt_2,141,14). coords(wk2gt_2,95,74,14).  visible(wk2gt_2,14). happensAt(walking(wk2gt_2),14).
orientation(wk2gt_1,143,15). coords(wk2gt_1,104,65,15).  visible(wk2gt_1,15). happensAt(walking(wk2gt_1),15). orientation(wk2gt_2,141,15). coords(wk2gt_2,95,73,15).  visible(wk2gt_2,15). happensAt(walking(wk2gt_2),15).
orientation(wk2gt_1,143,16). coords(wk2gt_1,104,65,16).  visible(wk2gt_1,16). happensAt(walking(wk2gt_1),16). orientation(wk2gt_2,141,16). coords(wk2gt_2,95,73,16).  visible(wk2gt_2,16). happensAt(walking(wk2gt_2),16).
orientation(wk2gt_1,143,17). coords(wk2gt_1,104,65,17).  visible(wk2gt_1,17). happensAt(walking(wk2gt_1),17). orientation(wk2gt_2,141,17). coords(wk2gt_2,95,73,17).  visible(wk2gt_2,17). happensAt(walking(wk2gt_2),17).
orientation(wk2gt_1,143,18). coords(wk2gt_1,105,64,18).  visible(wk2gt_1,18). happensAt(walking(wk2gt_1),18). orientation(wk2gt_2,141,18). coords(wk2gt_2,95,72,18).  visible(wk2gt_2,18). happensAt(walking(wk2gt_2),18).
orientation(wk2gt_1,143,19). coords(wk2gt_1,105,64,19).  visible(wk2gt_1,19). happensAt(walking(wk2gt_1),19). orientation(wk2gt_2,141,19). coords(wk2gt_2,96,71,19).  visible(wk2gt_2,19). happensAt(walking(wk2gt_2),19).
orientation(wk2gt_1,143,20). coords(wk2gt_1,105,64,20).  visible(wk2gt_1,20). happensAt(walking(wk2gt_1),20). orientation(wk2gt_2,141,20). coords(wk2gt_2,96,70,20).  visible(wk2gt_2,20). happensAt(walking(wk2gt_2),20).
orientation(wk2gt_1,143,21). coords(wk2gt_1,105,63,21).  visible(wk2gt_1,21). happensAt(walking(wk2gt_1),21). orientation(wk2gt_2,141,21). coords(wk2gt_2,96,70,21).  visible(wk2gt_2,21). happensAt(walking(wk2gt_2),21).
orientation(wk2gt_1,143,22). coords(wk2gt_1,105,62,22).  visible(wk2gt_1,22). happensAt(walking(wk2gt_1),22). orientation(wk2gt_2,141,22). coords(wk2gt_2,96,69,22).  visible(wk2gt_2,22). happensAt(walking(wk2gt_2),22).
orientation(wk2gt_1,143,23). coords(wk2gt_1,105,62,23).  visible(wk2gt_1,23). happensAt(walking(wk2gt_1),23). orientation(wk2gt_2,141,23). coords(wk2gt_2,97,68,23).  visible(wk2gt_2,23). happensAt(walking(wk2gt_2),23).
orientation(wk2gt_1,143,24). coords(wk2gt_1,105,62,24).  visible(wk2gt_1,24). happensAt(walking(wk2gt_1),24). orientation(wk2gt_2,141,24). coords(wk2gt_2,97,68,24).  visible(wk2gt_2,24). happensAt(walking(wk2gt_2),24).
orientation(wk2gt_1,143,25). coords(wk2gt_1,106,61,25).  visible(wk2gt_1,25). happensAt(walking(wk2gt_1),25). orientation(wk2gt_2,141,25). coords(wk2gt_2,97,67,25).  visible(wk2gt_2,25). happensAt(walking(wk2gt_2),25).
orientation(wk2gt_1,143,26). coords(wk2gt_1,106,60,26).  visible(wk2gt_1,26). happensAt(walking(wk2gt_1),26). orientation(wk2gt_2,141,26). coords(wk2gt_2,96,67,26).  visible(wk2gt_2,26). happensAt(walking(wk2gt_2),26).
orientation(wk2gt_1,143,27). coords(wk2gt_1,107,60,27).  visible(wk2gt_1,27). happensAt(walking(wk2gt_1),27). orientation(wk2gt_2,141,27). coords(wk2gt_2,97,67,27).  visible(wk2gt_2,27). happensAt(walking(wk2gt_2),27).
orientation(wk2gt_1,143,28). coords(wk2gt_1,108,60,28).  visible(wk2gt_1,28). happensAt(walking(wk2gt_1),28). orientation(wk2gt_2,141,28). coords(wk2gt_2,97,67,28).  visible(wk2gt_2,28). happensAt(walking(wk2gt_2),28).
orientation(wk2gt_1,143,29). coords(wk2gt_1,108,60,29).  visible(wk2gt_1,29). happensAt(walking(wk2gt_1),29). orientation(wk2gt_2,141,29). coords(wk2gt_2,97,66,29).  visible(wk2gt_2,29). happensAt(walking(wk2gt_2),29).
orientation(wk2gt_1,143,30). coords(wk2gt_1,108,60,30).  visible(wk2gt_1,30). happensAt(walking(wk2gt_1),30). orientation(wk2gt_2,141,30). coords(wk2gt_2,97,66,30).  visible(wk2gt_2,30). happensAt(walking(wk2gt_2),30).
orientation(wk2gt_1,143,31). coords(wk2gt_1,108,60,31).  visible(wk2gt_1,31). happensAt(walking(wk2gt_1),31). orientation(wk2gt_2,141,31). coords(wk2gt_2,97,66,31).  visible(wk2gt_2,31). happensAt(walking(wk2gt_2),31).
orientation(wk2gt_1,143,32). coords(wk2gt_1,108,60,32).  visible(wk2gt_1,32). happensAt(walking(wk2gt_1),32). orientation(wk2gt_2,141,32). coords(wk2gt_2,97,66,32).  visible(wk2gt_2,32). happensAt(walking(wk2gt_2),32).
orientation(wk2gt_1,143,33). coords(wk2gt_1,109,60,33).  visible(wk2gt_1,33). happensAt(walking(wk2gt_1),33). orientation(wk2gt_2,141,33). coords(wk2gt_2,97,66,33).  visible(wk2gt_2,33). happensAt(walking(wk2gt_2),33).
orientation(wk2gt_1,143,34). coords(wk2gt_1,109,59,34).  visible(wk2gt_1,34). happensAt(walking(wk2gt_1),34). orientation(wk2gt_2,141,34). coords(wk2gt_2,97,66,34).  visible(wk2gt_2,34). happensAt(walking(wk2gt_2),34).
orientation(wk2gt_1,143,35). coords(wk2gt_1,109,59,35).  visible(wk2gt_1,35). happensAt(walking(wk2gt_1),35). orientation(wk2gt_2,141,35). coords(wk2gt_2,97,66,35).  visible(wk2gt_2,35). happensAt(walking(wk2gt_2),35).
orientation(wk2gt_1,143,36). coords(wk2gt_1,109,58,36).  visible(wk2gt_1,36). happensAt(walking(wk2gt_1),36). orientation(wk2gt_2,141,36). coords(wk2gt_2,97,66,36).  visible(wk2gt_2,36). happensAt(walking(wk2gt_2),36).
#show holdsAt/2.