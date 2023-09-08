
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

	orientation(lb2gt_2,147,36). coords(lb2gt_2,20,53,36).  visible(lb2gt_2,36). happensAt(walking(lb2gt_2),36). orientation(lb2gt_3,146,36). coords(lb2gt_3,23,40,36).  visible(lb2gt_3,36). happensAt(walking(lb2gt_3),36).
orientation(lb2gt_2,147,37). coords(lb2gt_2,20,53,37).  visible(lb2gt_2,37). happensAt(walking(lb2gt_2),37). orientation(lb2gt_3,146,37). coords(lb2gt_3,23,40,37).  visible(lb2gt_3,37). happensAt(walking(lb2gt_3),37).
orientation(lb2gt_2,147,38). coords(lb2gt_2,20,54,38).  visible(lb2gt_2,38). happensAt(walking(lb2gt_2),38). orientation(lb2gt_3,146,38). coords(lb2gt_3,21,39,38).  visible(lb2gt_3,38). happensAt(walking(lb2gt_3),38).
orientation(lb2gt_2,147,39). coords(lb2gt_2,20,53,39).  visible(lb2gt_2,39). happensAt(walking(lb2gt_2),39). orientation(lb2gt_3,146,39). coords(lb2gt_3,21,40,39).  visible(lb2gt_3,39). happensAt(walking(lb2gt_3),39).
orientation(lb2gt_2,147,40). coords(lb2gt_2,20,53,40).  visible(lb2gt_2,40). happensAt(walking(lb2gt_2),40). orientation(lb2gt_3,146,40). coords(lb2gt_3,20,40,40).  visible(lb2gt_3,40). happensAt(walking(lb2gt_3),40).
orientation(lb2gt_2,147,41). coords(lb2gt_2,20,53,41).  visible(lb2gt_2,41). happensAt(walking(lb2gt_2),41). orientation(lb2gt_3,146,41). coords(lb2gt_3,20,40,41).  visible(lb2gt_3,41). happensAt(walking(lb2gt_3),41).
orientation(lb2gt_2,147,42). coords(lb2gt_2,20,54,42).  visible(lb2gt_2,42). happensAt(walking(lb2gt_2),42). orientation(lb2gt_3,146,42). coords(lb2gt_3,20,40,42).  visible(lb2gt_3,42). happensAt(walking(lb2gt_3),42).
orientation(lb2gt_2,147,43). coords(lb2gt_2,19,54,43).  visible(lb2gt_2,43). happensAt(walking(lb2gt_2),43). orientation(lb2gt_3,146,43). coords(lb2gt_3,20,40,43).  visible(lb2gt_3,43). happensAt(walking(lb2gt_3),43).
orientation(lb2gt_2,147,44). coords(lb2gt_2,19,54,44).  visible(lb2gt_2,44). happensAt(walking(lb2gt_2),44). orientation(lb2gt_3,146,44). coords(lb2gt_3,20,40,44).  visible(lb2gt_3,44). happensAt(walking(lb2gt_3),44).
orientation(lb2gt_2,147,45). coords(lb2gt_2,19,54,45).  visible(lb2gt_2,45). happensAt(walking(lb2gt_2),45). orientation(lb2gt_3,146,45). coords(lb2gt_3,20,40,45).  visible(lb2gt_3,45). happensAt(walking(lb2gt_3),45).
orientation(lb2gt_2,147,46). coords(lb2gt_2,19,54,46).  visible(lb2gt_2,46). happensAt(walking(lb2gt_2),46). orientation(lb2gt_3,146,46). coords(lb2gt_3,20,40,46).  visible(lb2gt_3,46). happensAt(walking(lb2gt_3),46).
orientation(lb2gt_2,147,47). coords(lb2gt_2,19,54,47).  visible(lb2gt_2,47). happensAt(walking(lb2gt_2),47). orientation(lb2gt_3,146,47). coords(lb2gt_3,20,40,47).  visible(lb2gt_3,47). happensAt(walking(lb2gt_3),47).
orientation(lb2gt_2,147,48). coords(lb2gt_2,19,54,48).  visible(lb2gt_2,48). happensAt(walking(lb2gt_2),48). orientation(lb2gt_3,146,48). coords(lb2gt_3,19,40,48).  visible(lb2gt_3,48). happensAt(walking(lb2gt_3),48).
orientation(lb2gt_2,147,49). coords(lb2gt_2,19,54,49).  visible(lb2gt_2,49). happensAt(walking(lb2gt_2),49). orientation(lb2gt_3,146,49). coords(lb2gt_3,19,41,49).  visible(lb2gt_3,49). happensAt(walking(lb2gt_3),49).
orientation(lb2gt_2,147,50). coords(lb2gt_2,19,54,50).  visible(lb2gt_2,50). happensAt(walking(lb2gt_2),50). orientation(lb2gt_3,146,50). coords(lb2gt_3,19,42,50).  visible(lb2gt_3,50). happensAt(walking(lb2gt_3),50).
orientation(lb2gt_2,147,51). coords(lb2gt_2,19,54,51).  visible(lb2gt_2,51). happensAt(walking(lb2gt_2),51). orientation(lb2gt_3,146,51). coords(lb2gt_3,18,42,51).  visible(lb2gt_3,51). happensAt(walking(lb2gt_3),51).
orientation(lb2gt_2,147,52). coords(lb2gt_2,19,54,52).  visible(lb2gt_2,52). happensAt(walking(lb2gt_2),52). orientation(lb2gt_3,146,52). coords(lb2gt_3,18,43,52).  visible(lb2gt_3,52). happensAt(walking(lb2gt_3),52).
orientation(lb2gt_2,147,53). coords(lb2gt_2,19,54,53).  visible(lb2gt_2,53). happensAt(walking(lb2gt_2),53). orientation(lb2gt_3,146,53). coords(lb2gt_3,18,43,53).  visible(lb2gt_3,53). happensAt(walking(lb2gt_3),53).
orientation(lb2gt_2,147,54). coords(lb2gt_2,19,54,54).  visible(lb2gt_2,54). happensAt(walking(lb2gt_2),54). orientation(lb2gt_3,146,54). coords(lb2gt_3,17,44,54).  visible(lb2gt_3,54). happensAt(walking(lb2gt_3),54).
orientation(lb2gt_2,147,55). coords(lb2gt_2,19,54,55).  visible(lb2gt_2,55). happensAt(walking(lb2gt_2),55). orientation(lb2gt_3,146,55). coords(lb2gt_3,17,45,55).  visible(lb2gt_3,55). happensAt(walking(lb2gt_3),55).
orientation(lb2gt_2,147,56). coords(lb2gt_2,19,54,56).  visible(lb2gt_2,56). happensAt(walking(lb2gt_2),56). orientation(lb2gt_3,146,56). coords(lb2gt_3,17,45,56).  visible(lb2gt_3,56). happensAt(walking(lb2gt_3),56).
orientation(lb2gt_2,147,57). coords(lb2gt_2,19,54,57).  visible(lb2gt_2,57). happensAt(walking(lb2gt_2),57). orientation(lb2gt_3,146,57). coords(lb2gt_3,17,45,57).  visible(lb2gt_3,57). happensAt(walking(lb2gt_3),57).
orientation(lb2gt_2,147,58). coords(lb2gt_2,19,54,58).  visible(lb2gt_2,58). happensAt(walking(lb2gt_2),58). orientation(lb2gt_3,146,58). coords(lb2gt_3,17,45,58).  visible(lb2gt_3,58). happensAt(walking(lb2gt_3),58).
orientation(lb2gt_2,147,59). coords(lb2gt_2,19,54,59).  visible(lb2gt_2,59). happensAt(walking(lb2gt_2),59). orientation(lb2gt_3,146,59). coords(lb2gt_3,18,45,59).  visible(lb2gt_3,59). happensAt(walking(lb2gt_3),59).
orientation(lb2gt_2,147,60). coords(lb2gt_2,19,54,60).  happensAt(disappear(lb2gt_2),60). happensAt(walking(lb2gt_2),60). orientation(lb2gt_3,146,60). coords(lb2gt_3,18,45,60).  visible(lb2gt_3,60). happensAt(walking(lb2gt_3),60).
#show holdsAt/2.