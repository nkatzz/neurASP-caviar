
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
orientation(mc1gt_0,148,65). coords(mc1gt_0,74,81,65).  visible(mc1gt_0,65). happensAt(walking(mc1gt_0),65). orientation(mc1gt_1,145,65). coords(mc1gt_1,82,78,65).  visible(mc1gt_1,65). happensAt(walking(mc1gt_1),65).
orientation(mc1gt_0,148,66). coords(mc1gt_0,75,81,66).  visible(mc1gt_0,66). happensAt(walking(mc1gt_0),66). orientation(mc1gt_1,145,66). coords(mc1gt_1,82,78,66).  visible(mc1gt_1,66). happensAt(walking(mc1gt_1),66).
orientation(mc1gt_0,148,67). coords(mc1gt_0,75,81,67).  visible(mc1gt_0,67). happensAt(walking(mc1gt_0),67). orientation(mc1gt_1,145,67). coords(mc1gt_1,83,79,67).  visible(mc1gt_1,67). happensAt(walking(mc1gt_1),67).
orientation(mc1gt_0,148,68). coords(mc1gt_0,75,82,68).  visible(mc1gt_0,68). happensAt(walking(mc1gt_0),68). orientation(mc1gt_1,145,68). coords(mc1gt_1,83,79,68).  visible(mc1gt_1,68). happensAt(walking(mc1gt_1),68).
orientation(mc1gt_0,148,69). coords(mc1gt_0,79,96,69).  visible(mc1gt_0,69). happensAt(walking(mc1gt_0),69). orientation(mc1gt_1,145,69). coords(mc1gt_1,94,83,69).  visible(mc1gt_1,69). happensAt(walking(mc1gt_1),69).
orientation(mc1gt_0,148,70). coords(mc1gt_0,79,96,70).  visible(mc1gt_0,70). happensAt(walking(mc1gt_0),70). orientation(mc1gt_1,145,70). coords(mc1gt_1,94,83,70).  visible(mc1gt_1,70). happensAt(walking(mc1gt_1),70).
orientation(mc1gt_0,148,71). coords(mc1gt_0,79,96,71).  visible(mc1gt_0,71). happensAt(walking(mc1gt_0),71). orientation(mc1gt_1,145,71). coords(mc1gt_1,94,83,71).  visible(mc1gt_1,71). happensAt(walking(mc1gt_1),71).
orientation(mc1gt_0,148,72). coords(mc1gt_0,79,96,72).  visible(mc1gt_0,72). happensAt(walking(mc1gt_0),72). orientation(mc1gt_1,145,72). coords(mc1gt_1,94,83,72).  visible(mc1gt_1,72). happensAt(walking(mc1gt_1),72).
orientation(mc1gt_0,148,73). coords(mc1gt_0,79,96,73).  visible(mc1gt_0,73). happensAt(walking(mc1gt_0),73). orientation(mc1gt_1,145,73). coords(mc1gt_1,94,83,73).  visible(mc1gt_1,73). happensAt(walking(mc1gt_1),73).
orientation(mc1gt_0,148,74). coords(mc1gt_0,79,96,74).  visible(mc1gt_0,74). happensAt(walking(mc1gt_0),74). orientation(mc1gt_1,145,74). coords(mc1gt_1,94,83,74).  visible(mc1gt_1,74). happensAt(walking(mc1gt_1),74).
orientation(mc1gt_0,148,75). coords(mc1gt_0,79,96,75).  visible(mc1gt_0,75). happensAt(walking(mc1gt_0),75). orientation(mc1gt_1,145,75). coords(mc1gt_1,94,83,75).  visible(mc1gt_1,75). happensAt(walking(mc1gt_1),75).
orientation(mc1gt_0,148,76). coords(mc1gt_0,79,96,76).  visible(mc1gt_0,76). happensAt(walking(mc1gt_0),76). orientation(mc1gt_1,145,76). coords(mc1gt_1,94,83,76).  visible(mc1gt_1,76). happensAt(walking(mc1gt_1),76).
#show holdsAt/2.