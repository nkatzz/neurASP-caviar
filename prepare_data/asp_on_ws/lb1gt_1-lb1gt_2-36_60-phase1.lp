
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
orientation(lb1gt_1,147,49). coords(lb1gt_1,73,92,49).  visible(lb1gt_1,49). happensAt(walking(lb1gt_1),49). orientation(lb1gt_2,142,49). coords(lb1gt_2,82,76,49).  visible(lb1gt_2,49). happensAt(walking(lb1gt_2),49).
orientation(lb1gt_1,147,50). coords(lb1gt_1,73,92,50).  visible(lb1gt_1,50). happensAt(walking(lb1gt_1),50). orientation(lb1gt_2,142,50). coords(lb1gt_2,82,76,50).  visible(lb1gt_2,50). happensAt(walking(lb1gt_2),50).
orientation(lb1gt_1,147,51). coords(lb1gt_1,73,93,51).  visible(lb1gt_1,51). happensAt(walking(lb1gt_1),51). orientation(lb1gt_2,142,51). coords(lb1gt_2,82,77,51).  visible(lb1gt_2,51). happensAt(walking(lb1gt_2),51).
orientation(lb1gt_1,147,52). coords(lb1gt_1,73,94,52).  visible(lb1gt_1,52). happensAt(walking(lb1gt_1),52). orientation(lb1gt_2,142,52). coords(lb1gt_2,82,78,52).  visible(lb1gt_2,52). happensAt(walking(lb1gt_2),52).
orientation(lb1gt_1,147,53). coords(lb1gt_1,73,95,53).  visible(lb1gt_1,53). happensAt(walking(lb1gt_1),53). orientation(lb1gt_2,142,53). coords(lb1gt_2,82,78,53).  visible(lb1gt_2,53). happensAt(walking(lb1gt_2),53).
orientation(lb1gt_1,147,54). coords(lb1gt_1,73,97,54).  visible(lb1gt_1,54). happensAt(walking(lb1gt_1),54). orientation(lb1gt_2,142,54). coords(lb1gt_2,82,78,54).  visible(lb1gt_2,54). happensAt(walking(lb1gt_2),54).
orientation(lb1gt_1,147,55). coords(lb1gt_1,73,98,55).  visible(lb1gt_1,55). happensAt(walking(lb1gt_1),55). orientation(lb1gt_2,142,55). coords(lb1gt_2,82,78,55).  visible(lb1gt_2,55). happensAt(walking(lb1gt_2),55).
orientation(lb1gt_1,147,56). coords(lb1gt_1,73,99,56).  visible(lb1gt_1,56). happensAt(walking(lb1gt_1),56). orientation(lb1gt_2,142,56). coords(lb1gt_2,82,79,56).  visible(lb1gt_2,56). happensAt(walking(lb1gt_2),56).
orientation(lb1gt_1,147,57). coords(lb1gt_1,73,100,57).  visible(lb1gt_1,57). happensAt(walking(lb1gt_1),57). orientation(lb1gt_2,142,57). coords(lb1gt_2,83,79,57).  visible(lb1gt_2,57). happensAt(walking(lb1gt_2),57).
orientation(lb1gt_1,147,58). coords(lb1gt_1,73,100,58).  visible(lb1gt_1,58). happensAt(walking(lb1gt_1),58). orientation(lb1gt_2,142,58). coords(lb1gt_2,83,79,58).  visible(lb1gt_2,58). happensAt(walking(lb1gt_2),58).
orientation(lb1gt_1,147,59). coords(lb1gt_1,73,100,59).  visible(lb1gt_1,59). happensAt(walking(lb1gt_1),59). orientation(lb1gt_2,142,59). coords(lb1gt_2,83,79,59).  visible(lb1gt_2,59). happensAt(walking(lb1gt_2),59).
orientation(lb1gt_1,147,60). coords(lb1gt_1,74,101,60).  visible(lb1gt_1,60). happensAt(walking(lb1gt_1),60). orientation(lb1gt_2,142,60). coords(lb1gt_2,83,79,60).  visible(lb1gt_2,60). happensAt(walking(lb1gt_2),60).
#show holdsAt/2.