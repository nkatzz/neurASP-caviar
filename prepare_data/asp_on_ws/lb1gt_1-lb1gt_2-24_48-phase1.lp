
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

	orientation(lb1gt_1,147,24). coords(lb1gt_1,71,86,24).  visible(lb1gt_1,24). happensAt(walking(lb1gt_1),24). orientation(lb1gt_2,142,24). coords(lb1gt_2,80,71,24).  visible(lb1gt_2,24). happensAt(walking(lb1gt_2),24).
orientation(lb1gt_1,147,25). coords(lb1gt_1,71,86,25).  visible(lb1gt_1,25). happensAt(walking(lb1gt_1),25). orientation(lb1gt_2,142,25). coords(lb1gt_2,80,72,25).  visible(lb1gt_2,25). happensAt(walking(lb1gt_2),25).
orientation(lb1gt_1,147,26). coords(lb1gt_1,71,87,26).  visible(lb1gt_1,26). happensAt(walking(lb1gt_1),26). orientation(lb1gt_2,142,26). coords(lb1gt_2,80,72,26).  visible(lb1gt_2,26). happensAt(walking(lb1gt_2),26).
orientation(lb1gt_1,147,27). coords(lb1gt_1,71,87,27).  visible(lb1gt_1,27). happensAt(walking(lb1gt_1),27). orientation(lb1gt_2,142,27). coords(lb1gt_2,80,73,27).  visible(lb1gt_2,27). happensAt(walking(lb1gt_2),27).
orientation(lb1gt_1,147,28). coords(lb1gt_1,71,87,28).  visible(lb1gt_1,28). happensAt(walking(lb1gt_1),28). orientation(lb1gt_2,142,28). coords(lb1gt_2,79,73,28).  visible(lb1gt_2,28). happensAt(walking(lb1gt_2),28).
orientation(lb1gt_1,147,29). coords(lb1gt_1,71,87,29).  visible(lb1gt_1,29). happensAt(walking(lb1gt_1),29). orientation(lb1gt_2,142,29). coords(lb1gt_2,79,74,29).  visible(lb1gt_2,29). happensAt(walking(lb1gt_2),29).
orientation(lb1gt_1,147,30). coords(lb1gt_1,71,88,30).  visible(lb1gt_1,30). happensAt(walking(lb1gt_1),30). orientation(lb1gt_2,142,30). coords(lb1gt_2,79,74,30).  visible(lb1gt_2,30). happensAt(walking(lb1gt_2),30).
orientation(lb1gt_1,147,31). coords(lb1gt_1,71,88,31).  visible(lb1gt_1,31). happensAt(walking(lb1gt_1),31). orientation(lb1gt_2,142,31). coords(lb1gt_2,79,74,31).  visible(lb1gt_2,31). happensAt(walking(lb1gt_2),31).
orientation(lb1gt_1,147,32). coords(lb1gt_1,72,88,32).  visible(lb1gt_1,32). happensAt(walking(lb1gt_1),32). orientation(lb1gt_2,142,32). coords(lb1gt_2,79,74,32).  visible(lb1gt_2,32). happensAt(walking(lb1gt_2),32).
orientation(lb1gt_1,147,33). coords(lb1gt_1,72,88,33).  visible(lb1gt_1,33). happensAt(walking(lb1gt_1),33). orientation(lb1gt_2,142,33). coords(lb1gt_2,79,74,33).  visible(lb1gt_2,33). happensAt(walking(lb1gt_2),33).
orientation(lb1gt_1,147,34). coords(lb1gt_1,72,88,34).  visible(lb1gt_1,34). happensAt(walking(lb1gt_1),34). orientation(lb1gt_2,142,34). coords(lb1gt_2,80,75,34).  visible(lb1gt_2,34). happensAt(walking(lb1gt_2),34).
orientation(lb1gt_1,147,35). coords(lb1gt_1,72,88,35).  visible(lb1gt_1,35). happensAt(walking(lb1gt_1),35). orientation(lb1gt_2,142,35). coords(lb1gt_2,80,75,35).  visible(lb1gt_2,35). happensAt(walking(lb1gt_2),35).
orientation(lb1gt_1,147,36). coords(lb1gt_1,72,88,36).  visible(lb1gt_1,36). happensAt(walking(lb1gt_1),36). orientation(lb1gt_2,142,36). coords(lb1gt_2,80,75,36).  visible(lb1gt_2,36). happensAt(walking(lb1gt_2),36).
orientation(lb1gt_1,147,37). coords(lb1gt_1,73,89,37).  visible(lb1gt_1,37). happensAt(walking(lb1gt_1),37). orientation(lb1gt_2,142,37). coords(lb1gt_2,80,75,37).  visible(lb1gt_2,37). happensAt(walking(lb1gt_2),37).
orientation(lb1gt_1,147,38). coords(lb1gt_1,73,89,38).  visible(lb1gt_1,38). happensAt(walking(lb1gt_1),38). orientation(lb1gt_2,142,38). coords(lb1gt_2,81,75,38).  visible(lb1gt_2,38). happensAt(walking(lb1gt_2),38).
orientation(lb1gt_1,147,39). coords(lb1gt_1,73,90,39).  visible(lb1gt_1,39). happensAt(walking(lb1gt_1),39). orientation(lb1gt_2,142,39). coords(lb1gt_2,81,75,39).  visible(lb1gt_2,39). happensAt(walking(lb1gt_2),39).
orientation(lb1gt_1,147,40). coords(lb1gt_1,73,90,40).  visible(lb1gt_1,40). happensAt(walking(lb1gt_1),40). orientation(lb1gt_2,142,40). coords(lb1gt_2,81,75,40).  visible(lb1gt_2,40). happensAt(walking(lb1gt_2),40).
orientation(lb1gt_1,147,41). coords(lb1gt_1,73,91,41).  visible(lb1gt_1,41). happensAt(walking(lb1gt_1),41). orientation(lb1gt_2,142,41). coords(lb1gt_2,81,76,41).  visible(lb1gt_2,41). happensAt(walking(lb1gt_2),41).
orientation(lb1gt_1,147,42). coords(lb1gt_1,73,91,42).  visible(lb1gt_1,42). happensAt(walking(lb1gt_1),42). orientation(lb1gt_2,142,42). coords(lb1gt_2,81,76,42).  visible(lb1gt_2,42). happensAt(walking(lb1gt_2),42).
orientation(lb1gt_1,147,43). coords(lb1gt_1,73,91,43).  visible(lb1gt_1,43). happensAt(walking(lb1gt_1),43). orientation(lb1gt_2,142,43). coords(lb1gt_2,82,76,43).  visible(lb1gt_2,43). happensAt(walking(lb1gt_2),43).
orientation(lb1gt_1,147,44). coords(lb1gt_1,73,91,44).  visible(lb1gt_1,44). happensAt(walking(lb1gt_1),44). orientation(lb1gt_2,142,44). coords(lb1gt_2,82,76,44).  visible(lb1gt_2,44). happensAt(walking(lb1gt_2),44).
orientation(lb1gt_1,147,45). coords(lb1gt_1,73,91,45).  visible(lb1gt_1,45). happensAt(walking(lb1gt_1),45). orientation(lb1gt_2,142,45). coords(lb1gt_2,82,76,45).  visible(lb1gt_2,45). happensAt(walking(lb1gt_2),45).
orientation(lb1gt_1,147,46). coords(lb1gt_1,73,92,46).  visible(lb1gt_1,46). happensAt(walking(lb1gt_1),46). orientation(lb1gt_2,142,46). coords(lb1gt_2,82,76,46).  visible(lb1gt_2,46). happensAt(walking(lb1gt_2),46).
orientation(lb1gt_1,147,47). coords(lb1gt_1,73,92,47).  visible(lb1gt_1,47). happensAt(walking(lb1gt_1),47). orientation(lb1gt_2,142,47). coords(lb1gt_2,82,76,47).  visible(lb1gt_2,47). happensAt(walking(lb1gt_2),47).
orientation(lb1gt_1,147,48). coords(lb1gt_1,73,92,48).  visible(lb1gt_1,48). happensAt(walking(lb1gt_1),48). orientation(lb1gt_2,142,48). coords(lb1gt_2,82,76,48).  visible(lb1gt_2,48). happensAt(walking(lb1gt_2),48).
#show holdsAt/2.