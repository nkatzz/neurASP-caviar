
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

	orientation(wk2gt_0,31,21). coords(wk2gt_0,211,261,21).  visible(wk2gt_0,21). happensAt(walking(wk2gt_0),21). orientation(wk2gt_2,141,21). coords(wk2gt_2,96,70,21).  visible(wk2gt_2,21). happensAt(walking(wk2gt_2),21).
orientation(wk2gt_0,31,22). coords(wk2gt_0,211,265,22).  visible(wk2gt_0,22). happensAt(walking(wk2gt_0),22). orientation(wk2gt_2,141,22). coords(wk2gt_2,96,69,22).  visible(wk2gt_2,22). happensAt(walking(wk2gt_2),22).
orientation(wk2gt_0,31,23). coords(wk2gt_0,212,268,23).  visible(wk2gt_0,23). happensAt(walking(wk2gt_0),23). orientation(wk2gt_2,141,23). coords(wk2gt_2,97,68,23).  visible(wk2gt_2,23). happensAt(walking(wk2gt_2),23).
orientation(wk2gt_0,31,24). coords(wk2gt_0,213,271,24).  visible(wk2gt_0,24). happensAt(walking(wk2gt_0),24). orientation(wk2gt_2,141,24). coords(wk2gt_2,97,68,24).  visible(wk2gt_2,24). happensAt(walking(wk2gt_2),24).
orientation(wk2gt_0,27,25). coords(wk2gt_0,213,273,25).  visible(wk2gt_0,25). happensAt(walking(wk2gt_0),25). orientation(wk2gt_2,141,25). coords(wk2gt_2,97,67,25).  visible(wk2gt_2,25). happensAt(walking(wk2gt_2),25).
orientation(wk2gt_0,27,26). coords(wk2gt_0,215,273,26).  visible(wk2gt_0,26). happensAt(walking(wk2gt_0),26). orientation(wk2gt_2,141,26). coords(wk2gt_2,96,67,26).  visible(wk2gt_2,26). happensAt(walking(wk2gt_2),26).
orientation(wk2gt_0,27,27). coords(wk2gt_0,216,273,27).  visible(wk2gt_0,27). happensAt(walking(wk2gt_0),27). orientation(wk2gt_2,141,27). coords(wk2gt_2,97,67,27).  visible(wk2gt_2,27). happensAt(walking(wk2gt_2),27).
orientation(wk2gt_0,32,28). coords(wk2gt_0,216,273,28).  visible(wk2gt_0,28). happensAt(walking(wk2gt_0),28). orientation(wk2gt_2,141,28). coords(wk2gt_2,97,67,28).  visible(wk2gt_2,28). happensAt(walking(wk2gt_2),28).
orientation(wk2gt_0,32,29). coords(wk2gt_0,217,273,29).  visible(wk2gt_0,29). happensAt(walking(wk2gt_0),29). orientation(wk2gt_2,141,29). coords(wk2gt_2,97,66,29).  visible(wk2gt_2,29). happensAt(walking(wk2gt_2),29).
orientation(wk2gt_0,39,30). coords(wk2gt_0,217,273,30).  visible(wk2gt_0,30). happensAt(walking(wk2gt_0),30). orientation(wk2gt_2,141,30). coords(wk2gt_2,97,66,30).  visible(wk2gt_2,30). happensAt(walking(wk2gt_2),30).
orientation(wk2gt_0,39,31). coords(wk2gt_0,218,274,31).  visible(wk2gt_0,31). happensAt(walking(wk2gt_0),31). orientation(wk2gt_2,141,31). coords(wk2gt_2,97,66,31).  visible(wk2gt_2,31). happensAt(walking(wk2gt_2),31).
orientation(wk2gt_0,39,32). coords(wk2gt_0,219,275,32).  visible(wk2gt_0,32). happensAt(walking(wk2gt_0),32). orientation(wk2gt_2,141,32). coords(wk2gt_2,97,66,32).  visible(wk2gt_2,32). happensAt(walking(wk2gt_2),32).
orientation(wk2gt_0,53,33). coords(wk2gt_0,222,277,33).  visible(wk2gt_0,33). happensAt(walking(wk2gt_0),33). orientation(wk2gt_2,141,33). coords(wk2gt_2,97,66,33).  visible(wk2gt_2,33). happensAt(walking(wk2gt_2),33).
orientation(wk2gt_0,53,34). coords(wk2gt_0,224,279,34).  visible(wk2gt_0,34). happensAt(walking(wk2gt_0),34). orientation(wk2gt_2,141,34). coords(wk2gt_2,97,66,34).  visible(wk2gt_2,34). happensAt(walking(wk2gt_2),34).
orientation(wk2gt_0,53,35). coords(wk2gt_0,224,280,35).  visible(wk2gt_0,35). happensAt(walking(wk2gt_0),35). orientation(wk2gt_2,141,35). coords(wk2gt_2,97,66,35).  visible(wk2gt_2,35). happensAt(walking(wk2gt_2),35).
orientation(wk2gt_0,70,36). coords(wk2gt_0,225,282,36).  visible(wk2gt_0,36). happensAt(walking(wk2gt_0),36). orientation(wk2gt_2,141,36). coords(wk2gt_2,97,66,36).  visible(wk2gt_2,36). happensAt(walking(wk2gt_2),36).
orientation(wk2gt_0,70,37). coords(wk2gt_0,231,283,37).  visible(wk2gt_0,37). happensAt(walking(wk2gt_0),37). orientation(wk2gt_2,141,37). coords(wk2gt_2,97,66,37).  visible(wk2gt_2,37). happensAt(walking(wk2gt_2),37).
orientation(wk2gt_0,70,38). coords(wk2gt_0,232,283,38).  visible(wk2gt_0,38). happensAt(walking(wk2gt_0),38). orientation(wk2gt_2,141,38). coords(wk2gt_2,97,66,38).  visible(wk2gt_2,38). happensAt(walking(wk2gt_2),38).
orientation(wk2gt_0,70,39). coords(wk2gt_0,232,283,39).  visible(wk2gt_0,39). happensAt(walking(wk2gt_0),39). orientation(wk2gt_2,141,39). coords(wk2gt_2,97,65,39).  visible(wk2gt_2,39). happensAt(walking(wk2gt_2),39).
orientation(wk2gt_0,70,40). coords(wk2gt_0,232,283,40).  visible(wk2gt_0,40). happensAt(walking(wk2gt_0),40). orientation(wk2gt_2,141,40). coords(wk2gt_2,98,65,40).  visible(wk2gt_2,40). happensAt(walking(wk2gt_2),40).
orientation(wk2gt_0,70,41). coords(wk2gt_0,232,283,41).  visible(wk2gt_0,41). happensAt(walking(wk2gt_0),41). orientation(wk2gt_2,141,41). coords(wk2gt_2,98,64,41).  visible(wk2gt_2,41). happensAt(walking(wk2gt_2),41).
orientation(wk2gt_0,70,42). coords(wk2gt_0,232,283,42).  visible(wk2gt_0,42). happensAt(walking(wk2gt_0),42). orientation(wk2gt_2,141,42). coords(wk2gt_2,98,64,42).  visible(wk2gt_2,42). happensAt(walking(wk2gt_2),42).
orientation(wk2gt_0,70,43). coords(wk2gt_0,233,284,43).  visible(wk2gt_0,43). happensAt(walking(wk2gt_0),43). orientation(wk2gt_2,141,43). coords(wk2gt_2,98,64,43).  visible(wk2gt_2,43). happensAt(walking(wk2gt_2),43).
orientation(wk2gt_0,70,44). coords(wk2gt_0,233,285,44).  visible(wk2gt_0,44). happensAt(walking(wk2gt_0),44). orientation(wk2gt_2,141,44). coords(wk2gt_2,98,63,44).  visible(wk2gt_2,44). happensAt(walking(wk2gt_2),44).
orientation(wk2gt_0,70,45). coords(wk2gt_0,233,286,45).  happensAt(disappear(wk2gt_0),45). happensAt(walking(wk2gt_0),45). orientation(wk2gt_2,141,45). coords(wk2gt_2,98,63,45).  visible(wk2gt_2,45). happensAt(walking(wk2gt_2),45).
#show holdsAt/2.