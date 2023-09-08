
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
orientation(lb2gt_0,141,37). coords(lb2gt_0,191,125,37).  visible(lb2gt_0,37). happensAt(active(lb2gt_0),37). orientation(lb2gt_2,147,37). coords(lb2gt_2,20,53,37).  visible(lb2gt_2,37). happensAt(walking(lb2gt_2),37).
orientation(lb2gt_0,141,38). coords(lb2gt_0,191,125,38).  visible(lb2gt_0,38). happensAt(active(lb2gt_0),38). orientation(lb2gt_2,147,38). coords(lb2gt_2,20,54,38).  visible(lb2gt_2,38). happensAt(walking(lb2gt_2),38).
orientation(lb2gt_0,141,39). coords(lb2gt_0,191,125,39).  visible(lb2gt_0,39). happensAt(active(lb2gt_0),39). orientation(lb2gt_2,147,39). coords(lb2gt_2,20,53,39).  visible(lb2gt_2,39). happensAt(walking(lb2gt_2),39).
orientation(lb2gt_0,141,40). coords(lb2gt_0,191,125,40).  visible(lb2gt_0,40). happensAt(active(lb2gt_0),40). orientation(lb2gt_2,147,40). coords(lb2gt_2,20,53,40).  visible(lb2gt_2,40). happensAt(walking(lb2gt_2),40).
orientation(lb2gt_0,141,41). coords(lb2gt_0,191,125,41).  visible(lb2gt_0,41). happensAt(active(lb2gt_0),41). orientation(lb2gt_2,147,41). coords(lb2gt_2,20,53,41).  visible(lb2gt_2,41). happensAt(walking(lb2gt_2),41).
orientation(lb2gt_0,141,42). coords(lb2gt_0,191,125,42).  visible(lb2gt_0,42). happensAt(active(lb2gt_0),42). orientation(lb2gt_2,147,42). coords(lb2gt_2,20,54,42).  visible(lb2gt_2,42). happensAt(walking(lb2gt_2),42).
orientation(lb2gt_0,141,43). coords(lb2gt_0,191,125,43).  visible(lb2gt_0,43). happensAt(active(lb2gt_0),43). orientation(lb2gt_2,147,43). coords(lb2gt_2,19,54,43).  visible(lb2gt_2,43). happensAt(walking(lb2gt_2),43).
orientation(lb2gt_0,141,44). coords(lb2gt_0,191,125,44).  visible(lb2gt_0,44). happensAt(active(lb2gt_0),44). orientation(lb2gt_2,147,44). coords(lb2gt_2,19,54,44).  visible(lb2gt_2,44). happensAt(walking(lb2gt_2),44).
orientation(lb2gt_0,141,45). coords(lb2gt_0,191,125,45).  visible(lb2gt_0,45). happensAt(active(lb2gt_0),45). orientation(lb2gt_2,147,45). coords(lb2gt_2,19,54,45).  visible(lb2gt_2,45). happensAt(walking(lb2gt_2),45).
orientation(lb2gt_0,141,46). coords(lb2gt_0,191,125,46).  visible(lb2gt_0,46). happensAt(active(lb2gt_0),46). orientation(lb2gt_2,147,46). coords(lb2gt_2,19,54,46).  visible(lb2gt_2,46). happensAt(walking(lb2gt_2),46).
orientation(lb2gt_0,141,47). coords(lb2gt_0,191,125,47).  visible(lb2gt_0,47). happensAt(active(lb2gt_0),47). orientation(lb2gt_2,147,47). coords(lb2gt_2,19,54,47).  visible(lb2gt_2,47). happensAt(walking(lb2gt_2),47).
orientation(lb2gt_0,141,48). coords(lb2gt_0,191,125,48).  visible(lb2gt_0,48). happensAt(active(lb2gt_0),48). orientation(lb2gt_2,147,48). coords(lb2gt_2,19,54,48).  visible(lb2gt_2,48). happensAt(walking(lb2gt_2),48).
#show holdsAt/2.