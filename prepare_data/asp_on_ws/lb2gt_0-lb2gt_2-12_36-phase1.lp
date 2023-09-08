
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

	orientation(lb2gt_0,141,12). coords(lb2gt_0,190,125,12).  visible(lb2gt_0,12). happensAt(inactive(lb2gt_0),12). orientation(lb2gt_2,147,12). coords(lb2gt_2,22,49,12).  visible(lb2gt_2,12). happensAt(walking(lb2gt_2),12).
orientation(lb2gt_0,141,13). coords(lb2gt_0,190,125,13).  visible(lb2gt_0,13). happensAt(inactive(lb2gt_0),13). orientation(lb2gt_2,147,13). coords(lb2gt_2,22,49,13).  visible(lb2gt_2,13). happensAt(walking(lb2gt_2),13).
orientation(lb2gt_0,141,14). coords(lb2gt_0,190,125,14).  visible(lb2gt_0,14). happensAt(inactive(lb2gt_0),14). orientation(lb2gt_2,147,14). coords(lb2gt_2,22,49,14).  visible(lb2gt_2,14). happensAt(walking(lb2gt_2),14).
orientation(lb2gt_0,141,15). coords(lb2gt_0,190,125,15).  visible(lb2gt_0,15). happensAt(inactive(lb2gt_0),15). orientation(lb2gt_2,147,15). coords(lb2gt_2,21,49,15).  visible(lb2gt_2,15). happensAt(walking(lb2gt_2),15).
orientation(lb2gt_0,141,16). coords(lb2gt_0,190,125,16).  visible(lb2gt_0,16). happensAt(inactive(lb2gt_0),16). orientation(lb2gt_2,147,16). coords(lb2gt_2,21,49,16).  visible(lb2gt_2,16). happensAt(walking(lb2gt_2),16).
orientation(lb2gt_0,141,17). coords(lb2gt_0,190,125,17).  visible(lb2gt_0,17). happensAt(inactive(lb2gt_0),17). orientation(lb2gt_2,147,17). coords(lb2gt_2,21,49,17).  visible(lb2gt_2,17). happensAt(walking(lb2gt_2),17).
orientation(lb2gt_0,141,18). coords(lb2gt_0,190,125,18).  visible(lb2gt_0,18). happensAt(inactive(lb2gt_0),18). orientation(lb2gt_2,147,18). coords(lb2gt_2,21,49,18).  visible(lb2gt_2,18). happensAt(walking(lb2gt_2),18).
orientation(lb2gt_0,141,19). coords(lb2gt_0,190,125,19).  visible(lb2gt_0,19). happensAt(inactive(lb2gt_0),19). orientation(lb2gt_2,147,19). coords(lb2gt_2,21,50,19).  visible(lb2gt_2,19). happensAt(walking(lb2gt_2),19).
orientation(lb2gt_0,141,20). coords(lb2gt_0,190,125,20).  visible(lb2gt_0,20). happensAt(inactive(lb2gt_0),20). orientation(lb2gt_2,147,20). coords(lb2gt_2,21,50,20).  visible(lb2gt_2,20). happensAt(walking(lb2gt_2),20).
orientation(lb2gt_0,141,21). coords(lb2gt_0,190,125,21).  visible(lb2gt_0,21). happensAt(inactive(lb2gt_0),21). orientation(lb2gt_2,147,21). coords(lb2gt_2,21,50,21).  visible(lb2gt_2,21). happensAt(walking(lb2gt_2),21).
orientation(lb2gt_0,141,22). coords(lb2gt_0,190,125,22).  visible(lb2gt_0,22). happensAt(inactive(lb2gt_0),22). orientation(lb2gt_2,147,22). coords(lb2gt_2,21,50,22).  visible(lb2gt_2,22). happensAt(walking(lb2gt_2),22).
orientation(lb2gt_0,141,23). coords(lb2gt_0,190,125,23).  visible(lb2gt_0,23). happensAt(inactive(lb2gt_0),23). orientation(lb2gt_2,147,23). coords(lb2gt_2,21,50,23).  visible(lb2gt_2,23). happensAt(walking(lb2gt_2),23).
orientation(lb2gt_0,141,24). coords(lb2gt_0,190,125,24).  visible(lb2gt_0,24). happensAt(inactive(lb2gt_0),24). orientation(lb2gt_2,147,24). coords(lb2gt_2,21,50,24).  visible(lb2gt_2,24). happensAt(walking(lb2gt_2),24).
orientation(lb2gt_0,141,25). coords(lb2gt_0,190,125,25).  visible(lb2gt_0,25). happensAt(active(lb2gt_0),25). orientation(lb2gt_2,147,25). coords(lb2gt_2,20,51,25).  visible(lb2gt_2,25). happensAt(walking(lb2gt_2),25).
orientation(lb2gt_0,141,26). coords(lb2gt_0,190,125,26).  visible(lb2gt_0,26). happensAt(active(lb2gt_0),26). orientation(lb2gt_2,147,26). coords(lb2gt_2,21,52,26).  visible(lb2gt_2,26). happensAt(walking(lb2gt_2),26).
orientation(lb2gt_0,141,27). coords(lb2gt_0,190,125,27).  visible(lb2gt_0,27). happensAt(active(lb2gt_0),27). orientation(lb2gt_2,147,27). coords(lb2gt_2,21,52,27).  visible(lb2gt_2,27). happensAt(walking(lb2gt_2),27).
orientation(lb2gt_0,141,28). coords(lb2gt_0,190,125,28).  visible(lb2gt_0,28). happensAt(active(lb2gt_0),28). orientation(lb2gt_2,147,28). coords(lb2gt_2,21,52,28).  visible(lb2gt_2,28). happensAt(walking(lb2gt_2),28).
orientation(lb2gt_0,141,29). coords(lb2gt_0,190,125,29).  visible(lb2gt_0,29). happensAt(active(lb2gt_0),29). orientation(lb2gt_2,147,29). coords(lb2gt_2,20,52,29).  visible(lb2gt_2,29). happensAt(walking(lb2gt_2),29).
orientation(lb2gt_0,141,30). coords(lb2gt_0,190,125,30).  visible(lb2gt_0,30). happensAt(active(lb2gt_0),30). orientation(lb2gt_2,147,30). coords(lb2gt_2,20,52,30).  visible(lb2gt_2,30). happensAt(walking(lb2gt_2),30).
orientation(lb2gt_0,141,31). coords(lb2gt_0,190,125,31).  visible(lb2gt_0,31). happensAt(active(lb2gt_0),31). orientation(lb2gt_2,147,31). coords(lb2gt_2,20,53,31).  visible(lb2gt_2,31). happensAt(walking(lb2gt_2),31).
orientation(lb2gt_0,141,32). coords(lb2gt_0,190,125,32).  visible(lb2gt_0,32). happensAt(active(lb2gt_0),32). orientation(lb2gt_2,147,32). coords(lb2gt_2,20,53,32).  visible(lb2gt_2,32). happensAt(walking(lb2gt_2),32).
orientation(lb2gt_0,141,33). coords(lb2gt_0,190,125,33).  visible(lb2gt_0,33). happensAt(active(lb2gt_0),33). orientation(lb2gt_2,147,33). coords(lb2gt_2,20,53,33).  visible(lb2gt_2,33). happensAt(walking(lb2gt_2),33).
orientation(lb2gt_0,141,34). coords(lb2gt_0,191,125,34).  visible(lb2gt_0,34). happensAt(active(lb2gt_0),34). orientation(lb2gt_2,147,34). coords(lb2gt_2,20,53,34).  visible(lb2gt_2,34). happensAt(walking(lb2gt_2),34).
orientation(lb2gt_0,141,35). coords(lb2gt_0,191,125,35).  visible(lb2gt_0,35). happensAt(active(lb2gt_0),35). orientation(lb2gt_2,147,35). coords(lb2gt_2,20,53,35).  visible(lb2gt_2,35). happensAt(walking(lb2gt_2),35).
orientation(lb2gt_0,141,36). coords(lb2gt_0,191,125,36).  visible(lb2gt_0,36). happensAt(active(lb2gt_0),36). orientation(lb2gt_2,147,36). coords(lb2gt_2,20,53,36).  visible(lb2gt_2,36). happensAt(walking(lb2gt_2),36).
#show holdsAt/2.