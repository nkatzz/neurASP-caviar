
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

	orientation(lb1gt_0,148,48). coords(lb1gt_0,182,119,48).  visible(lb1gt_0,48). happensAt(walking(lb1gt_0),48). orientation(lb1gt_2,142,48). coords(lb1gt_2,82,76,48).  visible(lb1gt_2,48). happensAt(walking(lb1gt_2),48).
orientation(lb1gt_0,148,49). coords(lb1gt_0,182,119,49).  visible(lb1gt_0,49). happensAt(walking(lb1gt_0),49). orientation(lb1gt_2,142,49). coords(lb1gt_2,82,76,49).  visible(lb1gt_2,49). happensAt(walking(lb1gt_2),49).
orientation(lb1gt_0,148,50). coords(lb1gt_0,182,119,50).  visible(lb1gt_0,50). happensAt(walking(lb1gt_0),50). orientation(lb1gt_2,142,50). coords(lb1gt_2,82,76,50).  visible(lb1gt_2,50). happensAt(walking(lb1gt_2),50).
orientation(lb1gt_0,148,51). coords(lb1gt_0,182,119,51).  visible(lb1gt_0,51). happensAt(walking(lb1gt_0),51). orientation(lb1gt_2,142,51). coords(lb1gt_2,82,77,51).  visible(lb1gt_2,51). happensAt(walking(lb1gt_2),51).
orientation(lb1gt_0,148,52). coords(lb1gt_0,182,119,52).  visible(lb1gt_0,52). happensAt(walking(lb1gt_0),52). orientation(lb1gt_2,142,52). coords(lb1gt_2,82,78,52).  visible(lb1gt_2,52). happensAt(walking(lb1gt_2),52).
orientation(lb1gt_0,148,53). coords(lb1gt_0,182,119,53).  visible(lb1gt_0,53). happensAt(walking(lb1gt_0),53). orientation(lb1gt_2,142,53). coords(lb1gt_2,82,78,53).  visible(lb1gt_2,53). happensAt(walking(lb1gt_2),53).
orientation(lb1gt_0,148,54). coords(lb1gt_0,182,119,54).  visible(lb1gt_0,54). happensAt(walking(lb1gt_0),54). orientation(lb1gt_2,142,54). coords(lb1gt_2,82,78,54).  visible(lb1gt_2,54). happensAt(walking(lb1gt_2),54).
orientation(lb1gt_0,148,55). coords(lb1gt_0,182,119,55).  visible(lb1gt_0,55). happensAt(walking(lb1gt_0),55). orientation(lb1gt_2,142,55). coords(lb1gt_2,82,78,55).  visible(lb1gt_2,55). happensAt(walking(lb1gt_2),55).
orientation(lb1gt_0,148,56). coords(lb1gt_0,181,119,56).  visible(lb1gt_0,56). happensAt(walking(lb1gt_0),56). orientation(lb1gt_2,142,56). coords(lb1gt_2,82,79,56).  visible(lb1gt_2,56). happensAt(walking(lb1gt_2),56).
orientation(lb1gt_0,148,57). coords(lb1gt_0,181,119,57).  visible(lb1gt_0,57). happensAt(active(lb1gt_0),57). orientation(lb1gt_2,142,57). coords(lb1gt_2,83,79,57).  visible(lb1gt_2,57). happensAt(walking(lb1gt_2),57).
orientation(lb1gt_0,148,58). coords(lb1gt_0,181,119,58).  visible(lb1gt_0,58). happensAt(active(lb1gt_0),58). orientation(lb1gt_2,142,58). coords(lb1gt_2,83,79,58).  visible(lb1gt_2,58). happensAt(walking(lb1gt_2),58).
orientation(lb1gt_0,148,59). coords(lb1gt_0,181,119,59).  visible(lb1gt_0,59). happensAt(active(lb1gt_0),59). orientation(lb1gt_2,142,59). coords(lb1gt_2,83,79,59).  visible(lb1gt_2,59). happensAt(walking(lb1gt_2),59).
orientation(lb1gt_0,148,60). coords(lb1gt_0,181,119,60).  visible(lb1gt_0,60). happensAt(active(lb1gt_0),60). orientation(lb1gt_2,142,60). coords(lb1gt_2,83,79,60).  visible(lb1gt_2,60). happensAt(walking(lb1gt_2),60).
orientation(lb1gt_0,148,61). coords(lb1gt_0,180,119,61).  visible(lb1gt_0,61). happensAt(active(lb1gt_0),61). orientation(lb1gt_2,142,61). coords(lb1gt_2,83,80,61).  visible(lb1gt_2,61). happensAt(walking(lb1gt_2),61).
orientation(lb1gt_0,148,62). coords(lb1gt_0,180,119,62).  visible(lb1gt_0,62). happensAt(active(lb1gt_0),62). orientation(lb1gt_2,142,62). coords(lb1gt_2,83,80,62).  visible(lb1gt_2,62). happensAt(walking(lb1gt_2),62).
orientation(lb1gt_0,148,63). coords(lb1gt_0,179,119,63).  visible(lb1gt_0,63). happensAt(active(lb1gt_0),63). orientation(lb1gt_2,142,63). coords(lb1gt_2,83,80,63).  visible(lb1gt_2,63). happensAt(walking(lb1gt_2),63).
orientation(lb1gt_0,148,64). coords(lb1gt_0,178,119,64).  visible(lb1gt_0,64). happensAt(active(lb1gt_0),64). orientation(lb1gt_2,142,64). coords(lb1gt_2,83,80,64).  visible(lb1gt_2,64). happensAt(walking(lb1gt_2),64).
orientation(lb1gt_0,148,65). coords(lb1gt_0,177,118,65).  visible(lb1gt_0,65). happensAt(active(lb1gt_0),65). orientation(lb1gt_2,142,65). coords(lb1gt_2,84,80,65).  visible(lb1gt_2,65). happensAt(walking(lb1gt_2),65).
orientation(lb1gt_0,148,66). coords(lb1gt_0,176,118,66).  visible(lb1gt_0,66). happensAt(active(lb1gt_0),66). orientation(lb1gt_2,142,66). coords(lb1gt_2,84,81,66).  visible(lb1gt_2,66). happensAt(walking(lb1gt_2),66).
orientation(lb1gt_0,148,67). coords(lb1gt_0,176,117,67).  visible(lb1gt_0,67). happensAt(active(lb1gt_0),67). orientation(lb1gt_2,142,67). coords(lb1gt_2,84,81,67).  visible(lb1gt_2,67). happensAt(walking(lb1gt_2),67).
orientation(lb1gt_0,148,68). coords(lb1gt_0,176,118,68).  visible(lb1gt_0,68). happensAt(active(lb1gt_0),68). orientation(lb1gt_2,142,68). coords(lb1gt_2,85,81,68).  visible(lb1gt_2,68). happensAt(walking(lb1gt_2),68).
orientation(lb1gt_0,148,69). coords(lb1gt_0,176,117,69).  visible(lb1gt_0,69). happensAt(active(lb1gt_0),69). orientation(lb1gt_2,142,69). coords(lb1gt_2,85,81,69).  visible(lb1gt_2,69). happensAt(walking(lb1gt_2),69).
orientation(lb1gt_0,148,70). coords(lb1gt_0,175,116,70).  visible(lb1gt_0,70). happensAt(active(lb1gt_0),70). orientation(lb1gt_2,142,70). coords(lb1gt_2,86,81,70).  visible(lb1gt_2,70). happensAt(walking(lb1gt_2),70).
orientation(lb1gt_0,148,71). coords(lb1gt_0,175,115,71).  visible(lb1gt_0,71). happensAt(active(lb1gt_0),71). orientation(lb1gt_2,142,71). coords(lb1gt_2,86,81,71).  visible(lb1gt_2,71). happensAt(walking(lb1gt_2),71).
orientation(lb1gt_0,148,72). coords(lb1gt_0,175,113,72).  visible(lb1gt_0,72). happensAt(active(lb1gt_0),72). orientation(lb1gt_2,142,72). coords(lb1gt_2,87,81,72).  visible(lb1gt_2,72). happensAt(walking(lb1gt_2),72).
#show holdsAt/2.