
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
orientation(wk2gt_1,143,37). coords(wk2gt_1,109,58,37).  visible(wk2gt_1,37). happensAt(walking(wk2gt_1),37). orientation(wk2gt_2,141,37). coords(wk2gt_2,97,66,37).  visible(wk2gt_2,37). happensAt(walking(wk2gt_2),37).
orientation(wk2gt_1,143,38). coords(wk2gt_1,109,57,38).  visible(wk2gt_1,38). happensAt(walking(wk2gt_1),38). orientation(wk2gt_2,141,38). coords(wk2gt_2,97,66,38).  visible(wk2gt_2,38). happensAt(walking(wk2gt_2),38).
orientation(wk2gt_1,143,39). coords(wk2gt_1,109,57,39).  visible(wk2gt_1,39). happensAt(walking(wk2gt_1),39). orientation(wk2gt_2,141,39). coords(wk2gt_2,97,65,39).  visible(wk2gt_2,39). happensAt(walking(wk2gt_2),39).
orientation(wk2gt_1,143,40). coords(wk2gt_1,109,56,40).  visible(wk2gt_1,40). happensAt(walking(wk2gt_1),40). orientation(wk2gt_2,141,40). coords(wk2gt_2,98,65,40).  visible(wk2gt_2,40). happensAt(walking(wk2gt_2),40).
orientation(wk2gt_1,143,41). coords(wk2gt_1,109,56,41).  visible(wk2gt_1,41). happensAt(walking(wk2gt_1),41). orientation(wk2gt_2,141,41). coords(wk2gt_2,98,64,41).  visible(wk2gt_2,41). happensAt(walking(wk2gt_2),41).
orientation(wk2gt_1,143,42). coords(wk2gt_1,109,57,42).  visible(wk2gt_1,42). happensAt(walking(wk2gt_1),42). orientation(wk2gt_2,141,42). coords(wk2gt_2,98,64,42).  visible(wk2gt_2,42). happensAt(walking(wk2gt_2),42).
orientation(wk2gt_1,143,43). coords(wk2gt_1,109,57,43).  visible(wk2gt_1,43). happensAt(walking(wk2gt_1),43). orientation(wk2gt_2,141,43). coords(wk2gt_2,98,64,43).  visible(wk2gt_2,43). happensAt(walking(wk2gt_2),43).
orientation(wk2gt_1,143,44). coords(wk2gt_1,109,57,44).  visible(wk2gt_1,44). happensAt(walking(wk2gt_1),44). orientation(wk2gt_2,141,44). coords(wk2gt_2,98,63,44).  visible(wk2gt_2,44). happensAt(walking(wk2gt_2),44).
orientation(wk2gt_1,143,45). coords(wk2gt_1,109,56,45).  visible(wk2gt_1,45). happensAt(walking(wk2gt_1),45). orientation(wk2gt_2,141,45). coords(wk2gt_2,98,63,45).  visible(wk2gt_2,45). happensAt(walking(wk2gt_2),45).
orientation(wk2gt_1,143,46). coords(wk2gt_1,109,56,46).  visible(wk2gt_1,46). happensAt(walking(wk2gt_1),46). orientation(wk2gt_2,141,46). coords(wk2gt_2,98,63,46).  visible(wk2gt_2,46). happensAt(walking(wk2gt_2),46).
orientation(wk2gt_1,143,47). coords(wk2gt_1,109,56,47).  visible(wk2gt_1,47). happensAt(walking(wk2gt_1),47). orientation(wk2gt_2,141,47). coords(wk2gt_2,98,63,47).  visible(wk2gt_2,47). happensAt(walking(wk2gt_2),47).
orientation(wk2gt_1,143,48). coords(wk2gt_1,109,56,48).  visible(wk2gt_1,48). happensAt(walking(wk2gt_1),48). orientation(wk2gt_2,141,48). coords(wk2gt_2,98,63,48).  visible(wk2gt_2,48). happensAt(walking(wk2gt_2),48).
#show holdsAt/2.