
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

	orientation(lb2gt_0,141,48). coords(lb2gt_0,191,125,48).  visible(lb2gt_0,48). happensAt(active(lb2gt_0),48). orientation(lb2gt_3,146,48). coords(lb2gt_3,19,40,48).  visible(lb2gt_3,48). happensAt(walking(lb2gt_3),48).
orientation(lb2gt_0,141,49). coords(lb2gt_0,191,125,49).  visible(lb2gt_0,49). happensAt(active(lb2gt_0),49). orientation(lb2gt_3,146,49). coords(lb2gt_3,19,41,49).  visible(lb2gt_3,49). happensAt(walking(lb2gt_3),49).
orientation(lb2gt_0,141,50). coords(lb2gt_0,191,125,50).  visible(lb2gt_0,50). happensAt(active(lb2gt_0),50). orientation(lb2gt_3,146,50). coords(lb2gt_3,19,42,50).  visible(lb2gt_3,50). happensAt(walking(lb2gt_3),50).
orientation(lb2gt_0,141,51). coords(lb2gt_0,191,125,51).  visible(lb2gt_0,51). happensAt(active(lb2gt_0),51). orientation(lb2gt_3,146,51). coords(lb2gt_3,18,42,51).  visible(lb2gt_3,51). happensAt(walking(lb2gt_3),51).
orientation(lb2gt_0,141,52). coords(lb2gt_0,191,125,52).  visible(lb2gt_0,52). happensAt(active(lb2gt_0),52). orientation(lb2gt_3,146,52). coords(lb2gt_3,18,43,52).  visible(lb2gt_3,52). happensAt(walking(lb2gt_3),52).
orientation(lb2gt_0,141,53). coords(lb2gt_0,191,125,53).  visible(lb2gt_0,53). happensAt(active(lb2gt_0),53). orientation(lb2gt_3,146,53). coords(lb2gt_3,18,43,53).  visible(lb2gt_3,53). happensAt(walking(lb2gt_3),53).
orientation(lb2gt_0,141,54). coords(lb2gt_0,191,125,54).  visible(lb2gt_0,54). happensAt(active(lb2gt_0),54). orientation(lb2gt_3,146,54). coords(lb2gt_3,17,44,54).  visible(lb2gt_3,54). happensAt(walking(lb2gt_3),54).
orientation(lb2gt_0,141,55). coords(lb2gt_0,191,125,55).  visible(lb2gt_0,55). happensAt(active(lb2gt_0),55). orientation(lb2gt_3,146,55). coords(lb2gt_3,17,45,55).  visible(lb2gt_3,55). happensAt(walking(lb2gt_3),55).
orientation(lb2gt_0,141,56). coords(lb2gt_0,191,125,56).  visible(lb2gt_0,56). happensAt(active(lb2gt_0),56). orientation(lb2gt_3,146,56). coords(lb2gt_3,17,45,56).  visible(lb2gt_3,56). happensAt(walking(lb2gt_3),56).
orientation(lb2gt_0,141,57). coords(lb2gt_0,192,125,57).  visible(lb2gt_0,57). happensAt(active(lb2gt_0),57). orientation(lb2gt_3,146,57). coords(lb2gt_3,17,45,57).  visible(lb2gt_3,57). happensAt(walking(lb2gt_3),57).
orientation(lb2gt_0,141,58). coords(lb2gt_0,192,125,58).  visible(lb2gt_0,58). happensAt(active(lb2gt_0),58). orientation(lb2gt_3,146,58). coords(lb2gt_3,17,45,58).  visible(lb2gt_3,58). happensAt(walking(lb2gt_3),58).
orientation(lb2gt_0,141,59). coords(lb2gt_0,193,125,59).  visible(lb2gt_0,59). happensAt(active(lb2gt_0),59). orientation(lb2gt_3,146,59). coords(lb2gt_3,18,45,59).  visible(lb2gt_3,59). happensAt(walking(lb2gt_3),59).
orientation(lb2gt_0,141,60). coords(lb2gt_0,193,124,60).  visible(lb2gt_0,60). happensAt(active(lb2gt_0),60). orientation(lb2gt_3,146,60). coords(lb2gt_3,18,45,60).  visible(lb2gt_3,60). happensAt(walking(lb2gt_3),60).
orientation(lb2gt_0,141,61). coords(lb2gt_0,193,124,61).  visible(lb2gt_0,61). happensAt(active(lb2gt_0),61). orientation(lb2gt_3,146,61). coords(lb2gt_3,18,45,61).  visible(lb2gt_3,61). happensAt(walking(lb2gt_3),61).
orientation(lb2gt_0,141,62). coords(lb2gt_0,193,124,62).  visible(lb2gt_0,62). happensAt(active(lb2gt_0),62). orientation(lb2gt_3,146,62). coords(lb2gt_3,17,46,62).  visible(lb2gt_3,62). happensAt(walking(lb2gt_3),62).
orientation(lb2gt_0,141,63). coords(lb2gt_0,193,124,63).  visible(lb2gt_0,63). happensAt(active(lb2gt_0),63). orientation(lb2gt_3,146,63). coords(lb2gt_3,17,46,63).  visible(lb2gt_3,63). happensAt(walking(lb2gt_3),63).
orientation(lb2gt_0,141,64). coords(lb2gt_0,193,124,64).  visible(lb2gt_0,64). happensAt(active(lb2gt_0),64). orientation(lb2gt_3,146,64). coords(lb2gt_3,17,46,64).  visible(lb2gt_3,64). happensAt(walking(lb2gt_3),64).
orientation(lb2gt_0,141,65). coords(lb2gt_0,193,124,65).  visible(lb2gt_0,65). happensAt(active(lb2gt_0),65). orientation(lb2gt_3,146,65). coords(lb2gt_3,17,46,65).  visible(lb2gt_3,65). happensAt(walking(lb2gt_3),65).
orientation(lb2gt_0,141,66). coords(lb2gt_0,193,124,66).  visible(lb2gt_0,66). happensAt(active(lb2gt_0),66). orientation(lb2gt_3,146,66). coords(lb2gt_3,16,46,66).  visible(lb2gt_3,66). happensAt(walking(lb2gt_3),66).
orientation(lb2gt_0,141,67). coords(lb2gt_0,193,124,67).  visible(lb2gt_0,67). happensAt(active(lb2gt_0),67). orientation(lb2gt_3,146,67). coords(lb2gt_3,16,46,67).  visible(lb2gt_3,67). happensAt(walking(lb2gt_3),67).
orientation(lb2gt_0,141,68). coords(lb2gt_0,193,124,68).  visible(lb2gt_0,68). happensAt(active(lb2gt_0),68). orientation(lb2gt_3,146,68). coords(lb2gt_3,15,46,68).  visible(lb2gt_3,68). happensAt(walking(lb2gt_3),68).
orientation(lb2gt_0,141,69). coords(lb2gt_0,193,124,69).  visible(lb2gt_0,69). happensAt(active(lb2gt_0),69). orientation(lb2gt_3,146,69). coords(lb2gt_3,15,46,69).  visible(lb2gt_3,69). happensAt(walking(lb2gt_3),69).
orientation(lb2gt_0,141,70). coords(lb2gt_0,193,124,70).  visible(lb2gt_0,70). happensAt(active(lb2gt_0),70). orientation(lb2gt_3,146,70). coords(lb2gt_3,15,46,70).  visible(lb2gt_3,70). happensAt(walking(lb2gt_3),70).
orientation(lb2gt_0,141,71). coords(lb2gt_0,193,124,71).  visible(lb2gt_0,71). happensAt(active(lb2gt_0),71). orientation(lb2gt_3,146,71). coords(lb2gt_3,15,46,71).  visible(lb2gt_3,71). happensAt(walking(lb2gt_3),71).
orientation(lb2gt_0,141,72). coords(lb2gt_0,193,124,72).  visible(lb2gt_0,72). happensAt(active(lb2gt_0),72). orientation(lb2gt_3,146,72). coords(lb2gt_3,14,46,72).  visible(lb2gt_3,72). happensAt(walking(lb2gt_3),72).
#show holdsAt/2.