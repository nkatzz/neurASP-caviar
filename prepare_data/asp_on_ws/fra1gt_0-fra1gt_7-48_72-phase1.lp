
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

	orientation(fra1gt_0,6,48). coords(fra1gt_0,232,198,48).  visible(fra1gt_0,48). happensAt(walking(fra1gt_0),48). orientation(fra1gt_7,136,48). coords(fra1gt_7,144,41,48).  visible(fra1gt_7,48). happensAt(inactive(fra1gt_7),48).
orientation(fra1gt_0,6,49). coords(fra1gt_0,232,201,49).  visible(fra1gt_0,49). happensAt(walking(fra1gt_0),49). orientation(fra1gt_7,136,49). coords(fra1gt_7,144,41,49).  visible(fra1gt_7,49). happensAt(inactive(fra1gt_7),49).
orientation(fra1gt_0,6,50). coords(fra1gt_0,233,205,50).  visible(fra1gt_0,50). happensAt(walking(fra1gt_0),50). orientation(fra1gt_7,136,50). coords(fra1gt_7,144,41,50).  visible(fra1gt_7,50). happensAt(inactive(fra1gt_7),50).
orientation(fra1gt_0,4,51). coords(fra1gt_0,233,208,51).  visible(fra1gt_0,51). happensAt(walking(fra1gt_0),51). orientation(fra1gt_7,136,51). coords(fra1gt_7,144,41,51).  visible(fra1gt_7,51). happensAt(inactive(fra1gt_7),51).
orientation(fra1gt_0,9,52). coords(fra1gt_0,234,211,52).  visible(fra1gt_0,52). happensAt(walking(fra1gt_0),52). orientation(fra1gt_7,136,52). coords(fra1gt_7,144,41,52).  visible(fra1gt_7,52). happensAt(inactive(fra1gt_7),52).
orientation(fra1gt_0,9,53). coords(fra1gt_0,235,213,53).  visible(fra1gt_0,53). happensAt(walking(fra1gt_0),53). orientation(fra1gt_7,136,53). coords(fra1gt_7,144,41,53).  visible(fra1gt_7,53). happensAt(inactive(fra1gt_7),53).
orientation(fra1gt_0,9,54). coords(fra1gt_0,236,216,54).  visible(fra1gt_0,54). happensAt(walking(fra1gt_0),54). orientation(fra1gt_7,136,54). coords(fra1gt_7,144,41,54).  visible(fra1gt_7,54). happensAt(inactive(fra1gt_7),54).
orientation(fra1gt_0,5,55). coords(fra1gt_0,238,218,55).  visible(fra1gt_0,55). happensAt(walking(fra1gt_0),55). orientation(fra1gt_7,136,55). coords(fra1gt_7,144,41,55).  visible(fra1gt_7,55). happensAt(inactive(fra1gt_7),55).
orientation(fra1gt_0,5,56). coords(fra1gt_0,240,221,56).  visible(fra1gt_0,56). happensAt(walking(fra1gt_0),56). orientation(fra1gt_7,136,56). coords(fra1gt_7,144,41,56).  visible(fra1gt_7,56). happensAt(inactive(fra1gt_7),56).
orientation(fra1gt_0,5,57). coords(fra1gt_0,243,223,57).  visible(fra1gt_0,57). happensAt(walking(fra1gt_0),57). orientation(fra1gt_7,136,57). coords(fra1gt_7,144,41,57).  visible(fra1gt_7,57). happensAt(inactive(fra1gt_7),57).
orientation(fra1gt_0,5,58). coords(fra1gt_0,244,225,58).  visible(fra1gt_0,58). happensAt(walking(fra1gt_0),58). orientation(fra1gt_7,136,58). coords(fra1gt_7,144,41,58).  visible(fra1gt_7,58). happensAt(inactive(fra1gt_7),58).
orientation(fra1gt_0,5,59). coords(fra1gt_0,247,226,59).  visible(fra1gt_0,59). happensAt(walking(fra1gt_0),59). orientation(fra1gt_7,136,59). coords(fra1gt_7,144,41,59).  visible(fra1gt_7,59). happensAt(inactive(fra1gt_7),59).
orientation(fra1gt_0,5,60). coords(fra1gt_0,247,227,60).  visible(fra1gt_0,60). happensAt(walking(fra1gt_0),60). orientation(fra1gt_7,136,60). coords(fra1gt_7,144,41,60).  visible(fra1gt_7,60). happensAt(inactive(fra1gt_7),60).
orientation(fra1gt_0,17,61). coords(fra1gt_0,250,230,61).  visible(fra1gt_0,61). happensAt(walking(fra1gt_0),61). orientation(fra1gt_7,136,61). coords(fra1gt_7,144,41,61).  visible(fra1gt_7,61). happensAt(inactive(fra1gt_7),61).
orientation(fra1gt_0,23,62). coords(fra1gt_0,250,233,62).  visible(fra1gt_0,62). happensAt(walking(fra1gt_0),62). orientation(fra1gt_7,136,62). coords(fra1gt_7,144,41,62).  visible(fra1gt_7,62). happensAt(inactive(fra1gt_7),62).
orientation(fra1gt_0,23,63). coords(fra1gt_0,251,236,63).  visible(fra1gt_0,63). happensAt(walking(fra1gt_0),63). orientation(fra1gt_7,136,63). coords(fra1gt_7,144,40,63).  visible(fra1gt_7,63). happensAt(inactive(fra1gt_7),63).
orientation(fra1gt_0,34,64). coords(fra1gt_0,251,240,64).  visible(fra1gt_0,64). happensAt(walking(fra1gt_0),64). orientation(fra1gt_7,136,64). coords(fra1gt_7,144,40,64).  visible(fra1gt_7,64). happensAt(inactive(fra1gt_7),64).
orientation(fra1gt_0,39,65). coords(fra1gt_0,251,244,65).  visible(fra1gt_0,65). happensAt(walking(fra1gt_0),65). orientation(fra1gt_7,136,65). coords(fra1gt_7,144,39,65).  visible(fra1gt_7,65). happensAt(inactive(fra1gt_7),65).
orientation(fra1gt_0,39,66). coords(fra1gt_0,254,247,66).  visible(fra1gt_0,66). happensAt(walking(fra1gt_0),66). orientation(fra1gt_7,136,66). coords(fra1gt_7,144,39,66).  visible(fra1gt_7,66). happensAt(inactive(fra1gt_7),66).
orientation(fra1gt_0,39,67). coords(fra1gt_0,258,251,67).  visible(fra1gt_0,67). happensAt(walking(fra1gt_0),67). orientation(fra1gt_7,136,67). coords(fra1gt_7,144,39,67).  visible(fra1gt_7,67). happensAt(inactive(fra1gt_7),67).
orientation(fra1gt_0,39,68). coords(fra1gt_0,261,253,68).  visible(fra1gt_0,68). happensAt(walking(fra1gt_0),68). orientation(fra1gt_7,136,68). coords(fra1gt_7,144,39,68).  visible(fra1gt_7,68). happensAt(inactive(fra1gt_7),68).
orientation(fra1gt_0,39,69). coords(fra1gt_0,264,254,69).  visible(fra1gt_0,69). happensAt(walking(fra1gt_0),69). orientation(fra1gt_7,136,69). coords(fra1gt_7,144,39,69).  visible(fra1gt_7,69). happensAt(inactive(fra1gt_7),69).
orientation(fra1gt_0,39,70). coords(fra1gt_0,266,255,70).  visible(fra1gt_0,70). happensAt(walking(fra1gt_0),70). orientation(fra1gt_7,136,70). coords(fra1gt_7,144,39,70).  visible(fra1gt_7,70). happensAt(inactive(fra1gt_7),70).
orientation(fra1gt_0,39,71). coords(fra1gt_0,267,257,71).  visible(fra1gt_0,71). happensAt(walking(fra1gt_0),71). orientation(fra1gt_7,136,71). coords(fra1gt_7,144,39,71).  visible(fra1gt_7,71). happensAt(inactive(fra1gt_7),71).
orientation(fra1gt_0,39,72). coords(fra1gt_0,269,258,72).  visible(fra1gt_0,72). happensAt(walking(fra1gt_0),72). orientation(fra1gt_7,136,72). coords(fra1gt_7,144,39,72).  visible(fra1gt_7,72). happensAt(inactive(fra1gt_7),72).
#show holdsAt/2.