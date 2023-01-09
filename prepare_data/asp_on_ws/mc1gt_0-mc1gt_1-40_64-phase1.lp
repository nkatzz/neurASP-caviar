
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

	orientation(mc1gt_0,148,40). coords(mc1gt_0,78,67,40).  visible(mc1gt_0,40). happensAt(walking(mc1gt_0),40). orientation(mc1gt_1,145,40). coords(mc1gt_1,83,66,40).  happensAt(appear(mc1gt_1),40). happensAt(walking(mc1gt_1),40).
orientation(mc1gt_0,148,41). coords(mc1gt_0,78,68,41).  visible(mc1gt_0,41). happensAt(walking(mc1gt_0),41). orientation(mc1gt_1,145,41). coords(mc1gt_1,83,66,41).  visible(mc1gt_1,41). happensAt(walking(mc1gt_1),41).
orientation(mc1gt_0,148,42). coords(mc1gt_0,78,70,42).  visible(mc1gt_0,42). happensAt(walking(mc1gt_0),42). orientation(mc1gt_1,145,42). coords(mc1gt_1,83,66,42).  visible(mc1gt_1,42). happensAt(walking(mc1gt_1),42).
orientation(mc1gt_0,148,43). coords(mc1gt_0,78,70,43).  visible(mc1gt_0,43). happensAt(walking(mc1gt_0),43). orientation(mc1gt_1,145,43). coords(mc1gt_1,83,66,43).  visible(mc1gt_1,43). happensAt(walking(mc1gt_1),43).
orientation(mc1gt_0,148,44). coords(mc1gt_0,78,71,44).  visible(mc1gt_0,44). happensAt(walking(mc1gt_0),44). orientation(mc1gt_1,145,44). coords(mc1gt_1,83,67,44).  visible(mc1gt_1,44). happensAt(walking(mc1gt_1),44).
orientation(mc1gt_0,148,45). coords(mc1gt_0,78,71,45).  visible(mc1gt_0,45). happensAt(walking(mc1gt_0),45). orientation(mc1gt_1,145,45). coords(mc1gt_1,83,68,45).  visible(mc1gt_1,45). happensAt(walking(mc1gt_1),45).
orientation(mc1gt_0,148,46). coords(mc1gt_0,77,72,46).  visible(mc1gt_0,46). happensAt(walking(mc1gt_0),46). orientation(mc1gt_1,145,46). coords(mc1gt_1,83,69,46).  visible(mc1gt_1,46). happensAt(walking(mc1gt_1),46).
orientation(mc1gt_0,148,47). coords(mc1gt_0,77,72,47).  visible(mc1gt_0,47). happensAt(walking(mc1gt_0),47). orientation(mc1gt_1,145,47). coords(mc1gt_1,83,71,47).  visible(mc1gt_1,47). happensAt(walking(mc1gt_1),47).
orientation(mc1gt_0,148,48). coords(mc1gt_0,77,72,48).  visible(mc1gt_0,48). happensAt(walking(mc1gt_0),48). orientation(mc1gt_1,145,48). coords(mc1gt_1,83,72,48).  visible(mc1gt_1,48). happensAt(walking(mc1gt_1),48).
orientation(mc1gt_0,148,49). coords(mc1gt_0,77,72,49).  visible(mc1gt_0,49). happensAt(walking(mc1gt_0),49). orientation(mc1gt_1,145,49). coords(mc1gt_1,83,72,49).  visible(mc1gt_1,49). happensAt(walking(mc1gt_1),49).
orientation(mc1gt_0,148,50). coords(mc1gt_0,77,73,50).  visible(mc1gt_0,50). happensAt(walking(mc1gt_0),50). orientation(mc1gt_1,145,50). coords(mc1gt_1,82,73,50).  visible(mc1gt_1,50). happensAt(walking(mc1gt_1),50).
orientation(mc1gt_0,148,51). coords(mc1gt_0,77,73,51).  visible(mc1gt_0,51). happensAt(walking(mc1gt_0),51). orientation(mc1gt_1,145,51). coords(mc1gt_1,82,73,51).  visible(mc1gt_1,51). happensAt(walking(mc1gt_1),51).
orientation(mc1gt_0,148,52). coords(mc1gt_0,77,73,52).  visible(mc1gt_0,52). happensAt(walking(mc1gt_0),52). orientation(mc1gt_1,145,52). coords(mc1gt_1,82,73,52).  visible(mc1gt_1,52). happensAt(walking(mc1gt_1),52).
orientation(mc1gt_0,148,53). coords(mc1gt_0,76,74,53).  visible(mc1gt_0,53). happensAt(walking(mc1gt_0),53). orientation(mc1gt_1,145,53). coords(mc1gt_1,82,74,53).  visible(mc1gt_1,53). happensAt(walking(mc1gt_1),53).
orientation(mc1gt_0,148,54). coords(mc1gt_0,75,76,54).  visible(mc1gt_0,54). happensAt(walking(mc1gt_0),54). orientation(mc1gt_1,145,54). coords(mc1gt_1,81,74,54).  visible(mc1gt_1,54). happensAt(walking(mc1gt_1),54).
orientation(mc1gt_0,148,55). coords(mc1gt_0,75,76,55).  visible(mc1gt_0,55). happensAt(walking(mc1gt_0),55). orientation(mc1gt_1,145,55). coords(mc1gt_1,81,74,55).  visible(mc1gt_1,55). happensAt(walking(mc1gt_1),55).
orientation(mc1gt_0,148,56). coords(mc1gt_0,75,78,56).  visible(mc1gt_0,56). happensAt(walking(mc1gt_0),56). orientation(mc1gt_1,145,56). coords(mc1gt_1,81,74,56).  visible(mc1gt_1,56). happensAt(walking(mc1gt_1),56).
orientation(mc1gt_0,148,57). coords(mc1gt_0,75,78,57).  visible(mc1gt_0,57). happensAt(walking(mc1gt_0),57). orientation(mc1gt_1,145,57). coords(mc1gt_1,81,75,57).  visible(mc1gt_1,57). happensAt(walking(mc1gt_1),57).
orientation(mc1gt_0,148,58). coords(mc1gt_0,75,79,58).  visible(mc1gt_0,58). happensAt(walking(mc1gt_0),58). orientation(mc1gt_1,145,58). coords(mc1gt_1,81,74,58).  visible(mc1gt_1,58). happensAt(walking(mc1gt_1),58).
orientation(mc1gt_0,148,59). coords(mc1gt_0,75,79,59).  visible(mc1gt_0,59). happensAt(walking(mc1gt_0),59). orientation(mc1gt_1,145,59). coords(mc1gt_1,81,75,59).  visible(mc1gt_1,59). happensAt(walking(mc1gt_1),59).
orientation(mc1gt_0,148,60). coords(mc1gt_0,75,80,60).  visible(mc1gt_0,60). happensAt(walking(mc1gt_0),60). orientation(mc1gt_1,145,60). coords(mc1gt_1,81,75,60).  visible(mc1gt_1,60). happensAt(walking(mc1gt_1),60).
orientation(mc1gt_0,148,61). coords(mc1gt_0,75,80,61).  visible(mc1gt_0,61). happensAt(walking(mc1gt_0),61). orientation(mc1gt_1,145,61). coords(mc1gt_1,81,76,61).  visible(mc1gt_1,61). happensAt(walking(mc1gt_1),61).
orientation(mc1gt_0,148,62). coords(mc1gt_0,75,80,62).  visible(mc1gt_0,62). happensAt(walking(mc1gt_0),62). orientation(mc1gt_1,145,62). coords(mc1gt_1,82,76,62).  visible(mc1gt_1,62). happensAt(walking(mc1gt_1),62).
orientation(mc1gt_0,148,63). coords(mc1gt_0,75,81,63).  visible(mc1gt_0,63). happensAt(walking(mc1gt_0),63). orientation(mc1gt_1,145,63). coords(mc1gt_1,82,77,63).  visible(mc1gt_1,63). happensAt(walking(mc1gt_1),63).
orientation(mc1gt_0,148,64). coords(mc1gt_0,75,81,64).  visible(mc1gt_0,64). happensAt(walking(mc1gt_0),64). orientation(mc1gt_1,145,64). coords(mc1gt_1,82,77,64).  visible(mc1gt_1,64). happensAt(walking(mc1gt_1),64).
#show holdsAt/2.