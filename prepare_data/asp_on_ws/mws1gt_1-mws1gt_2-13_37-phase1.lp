
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

	orientation(mws1gt_1,170,13). coords(mws1gt_1,65,193,13).  visible(mws1gt_1,13). happensAt(inactive(mws1gt_1),13). orientation(mws1gt_2,144,13). coords(mws1gt_2,46,114,13).  visible(mws1gt_2,13). happensAt(walking(mws1gt_2),13).
orientation(mws1gt_1,170,14). coords(mws1gt_1,65,193,14).  visible(mws1gt_1,14). happensAt(inactive(mws1gt_1),14). orientation(mws1gt_2,144,14). coords(mws1gt_2,46,115,14).  visible(mws1gt_2,14). happensAt(walking(mws1gt_2),14).
orientation(mws1gt_1,170,15). coords(mws1gt_1,65,193,15).  visible(mws1gt_1,15). happensAt(inactive(mws1gt_1),15). orientation(mws1gt_2,144,15). coords(mws1gt_2,47,116,15).  visible(mws1gt_2,15). happensAt(walking(mws1gt_2),15).
orientation(mws1gt_1,170,16). coords(mws1gt_1,65,193,16).  visible(mws1gt_1,16). happensAt(inactive(mws1gt_1),16). orientation(mws1gt_2,144,16). coords(mws1gt_2,47,117,16).  visible(mws1gt_2,16). happensAt(walking(mws1gt_2),16).
orientation(mws1gt_1,170,17). coords(mws1gt_1,65,193,17).  visible(mws1gt_1,17). happensAt(inactive(mws1gt_1),17). orientation(mws1gt_2,144,17). coords(mws1gt_2,47,118,17).  visible(mws1gt_2,17). happensAt(walking(mws1gt_2),17).
orientation(mws1gt_1,170,18). coords(mws1gt_1,65,193,18).  visible(mws1gt_1,18). happensAt(inactive(mws1gt_1),18). orientation(mws1gt_2,155,18). coords(mws1gt_2,46,120,18).  visible(mws1gt_2,18). happensAt(walking(mws1gt_2),18).
orientation(mws1gt_1,170,19). coords(mws1gt_1,65,193,19).  visible(mws1gt_1,19). happensAt(inactive(mws1gt_1),19). orientation(mws1gt_2,155,19). coords(mws1gt_2,46,120,19).  visible(mws1gt_2,19). happensAt(walking(mws1gt_2),19).
orientation(mws1gt_1,170,20). coords(mws1gt_1,65,193,20).  visible(mws1gt_1,20). happensAt(inactive(mws1gt_1),20). orientation(mws1gt_2,155,20). coords(mws1gt_2,46,121,20).  visible(mws1gt_2,20). happensAt(walking(mws1gt_2),20).
orientation(mws1gt_1,170,21). coords(mws1gt_1,65,193,21).  visible(mws1gt_1,21). happensAt(inactive(mws1gt_1),21). orientation(mws1gt_2,155,21). coords(mws1gt_2,46,122,21).  visible(mws1gt_2,21). happensAt(walking(mws1gt_2),21).
orientation(mws1gt_1,170,22). coords(mws1gt_1,65,193,22).  visible(mws1gt_1,22). happensAt(inactive(mws1gt_1),22). orientation(mws1gt_2,155,22). coords(mws1gt_2,44,123,22).  visible(mws1gt_2,22). happensAt(walking(mws1gt_2),22).
orientation(mws1gt_1,170,23). coords(mws1gt_1,65,193,23).  visible(mws1gt_1,23). happensAt(inactive(mws1gt_1),23). orientation(mws1gt_2,155,23). coords(mws1gt_2,44,124,23).  visible(mws1gt_2,23). happensAt(walking(mws1gt_2),23).
orientation(mws1gt_1,170,24). coords(mws1gt_1,65,193,24).  visible(mws1gt_1,24). happensAt(inactive(mws1gt_1),24). orientation(mws1gt_2,155,24). coords(mws1gt_2,44,125,24).  visible(mws1gt_2,24). happensAt(walking(mws1gt_2),24).
orientation(mws1gt_1,170,25). coords(mws1gt_1,65,193,25).  visible(mws1gt_1,25). happensAt(inactive(mws1gt_1),25). orientation(mws1gt_2,155,25). coords(mws1gt_2,44,126,25).  visible(mws1gt_2,25). happensAt(walking(mws1gt_2),25).
orientation(mws1gt_1,170,26). coords(mws1gt_1,65,193,26).  visible(mws1gt_1,26). happensAt(inactive(mws1gt_1),26). orientation(mws1gt_2,155,26). coords(mws1gt_2,44,126,26).  visible(mws1gt_2,26). happensAt(walking(mws1gt_2),26).
orientation(mws1gt_1,170,27). coords(mws1gt_1,65,193,27).  visible(mws1gt_1,27). happensAt(inactive(mws1gt_1),27). orientation(mws1gt_2,155,27). coords(mws1gt_2,44,127,27).  visible(mws1gt_2,27). happensAt(walking(mws1gt_2),27).
orientation(mws1gt_1,170,28). coords(mws1gt_1,65,193,28).  visible(mws1gt_1,28). happensAt(inactive(mws1gt_1),28). orientation(mws1gt_2,155,28). coords(mws1gt_2,44,128,28).  visible(mws1gt_2,28). happensAt(walking(mws1gt_2),28).
orientation(mws1gt_1,170,29). coords(mws1gt_1,65,193,29).  visible(mws1gt_1,29). happensAt(inactive(mws1gt_1),29). orientation(mws1gt_2,155,29). coords(mws1gt_2,43,129,29).  visible(mws1gt_2,29). happensAt(walking(mws1gt_2),29).
orientation(mws1gt_1,170,30). coords(mws1gt_1,65,193,30).  visible(mws1gt_1,30). happensAt(inactive(mws1gt_1),30). orientation(mws1gt_2,155,30). coords(mws1gt_2,44,130,30).  visible(mws1gt_2,30). happensAt(walking(mws1gt_2),30).
orientation(mws1gt_1,170,31). coords(mws1gt_1,65,193,31).  visible(mws1gt_1,31). happensAt(inactive(mws1gt_1),31). orientation(mws1gt_2,155,31). coords(mws1gt_2,44,130,31).  visible(mws1gt_2,31). happensAt(walking(mws1gt_2),31).
orientation(mws1gt_1,170,32). coords(mws1gt_1,65,193,32).  visible(mws1gt_1,32). happensAt(inactive(mws1gt_1),32). orientation(mws1gt_2,155,32). coords(mws1gt_2,44,131,32).  visible(mws1gt_2,32). happensAt(walking(mws1gt_2),32).
orientation(mws1gt_1,170,33). coords(mws1gt_1,65,193,33).  visible(mws1gt_1,33). happensAt(inactive(mws1gt_1),33). orientation(mws1gt_2,155,33). coords(mws1gt_2,44,132,33).  visible(mws1gt_2,33). happensAt(walking(mws1gt_2),33).
orientation(mws1gt_1,170,34). coords(mws1gt_1,65,193,34).  visible(mws1gt_1,34). happensAt(inactive(mws1gt_1),34). orientation(mws1gt_2,155,34). coords(mws1gt_2,44,133,34).  visible(mws1gt_2,34). happensAt(walking(mws1gt_2),34).
orientation(mws1gt_1,170,35). coords(mws1gt_1,65,193,35).  visible(mws1gt_1,35). happensAt(inactive(mws1gt_1),35). orientation(mws1gt_2,155,35). coords(mws1gt_2,43,134,35).  visible(mws1gt_2,35). happensAt(walking(mws1gt_2),35).
orientation(mws1gt_1,170,36). coords(mws1gt_1,65,193,36).  visible(mws1gt_1,36). happensAt(inactive(mws1gt_1),36). orientation(mws1gt_2,155,36). coords(mws1gt_2,44,135,36).  visible(mws1gt_2,36). happensAt(walking(mws1gt_2),36).
orientation(mws1gt_1,170,37). coords(mws1gt_1,65,193,37).  visible(mws1gt_1,37). happensAt(inactive(mws1gt_1),37). orientation(mws1gt_2,155,37). coords(mws1gt_2,44,136,37).  visible(mws1gt_2,37). happensAt(walking(mws1gt_2),37).
#show holdsAt/2.