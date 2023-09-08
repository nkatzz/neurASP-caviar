
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

	orientation(fcgt_0,143,48). coords(fcgt_0,190,145,48).  visible(fcgt_0,48). happensAt(walking(fcgt_0),48). orientation(fcgt_3,159,48). coords(fcgt_3,103,163,48).  visible(fcgt_3,48). happensAt(inactive(fcgt_3),48).
orientation(fcgt_0,143,49). coords(fcgt_0,191,146,49).  visible(fcgt_0,49). happensAt(walking(fcgt_0),49). orientation(fcgt_3,159,49). coords(fcgt_3,103,163,49).  visible(fcgt_3,49). happensAt(inactive(fcgt_3),49).
orientation(fcgt_0,143,50). coords(fcgt_0,192,147,50).  visible(fcgt_0,50). happensAt(walking(fcgt_0),50). orientation(fcgt_3,159,50). coords(fcgt_3,103,163,50).  visible(fcgt_3,50). happensAt(inactive(fcgt_3),50).
orientation(fcgt_0,143,51). coords(fcgt_0,192,147,51).  visible(fcgt_0,51). happensAt(walking(fcgt_0),51). orientation(fcgt_3,159,51). coords(fcgt_3,103,163,51).  visible(fcgt_3,51). happensAt(inactive(fcgt_3),51).
orientation(fcgt_0,143,52). coords(fcgt_0,192,147,52).  visible(fcgt_0,52). happensAt(walking(fcgt_0),52). orientation(fcgt_3,159,52). coords(fcgt_3,103,163,52).  visible(fcgt_3,52). happensAt(inactive(fcgt_3),52).
orientation(fcgt_0,143,53). coords(fcgt_0,193,148,53).  visible(fcgt_0,53). happensAt(walking(fcgt_0),53). orientation(fcgt_3,159,53). coords(fcgt_3,103,163,53).  visible(fcgt_3,53). happensAt(inactive(fcgt_3),53).
orientation(fcgt_0,143,54). coords(fcgt_0,193,148,54).  visible(fcgt_0,54). happensAt(walking(fcgt_0),54). orientation(fcgt_3,159,54). coords(fcgt_3,103,163,54).  visible(fcgt_3,54). happensAt(inactive(fcgt_3),54).
orientation(fcgt_0,143,55). coords(fcgt_0,193,148,55).  visible(fcgt_0,55). happensAt(walking(fcgt_0),55). orientation(fcgt_3,159,55). coords(fcgt_3,103,163,55).  visible(fcgt_3,55). happensAt(inactive(fcgt_3),55).
orientation(fcgt_0,143,56). coords(fcgt_0,194,148,56).  visible(fcgt_0,56). happensAt(walking(fcgt_0),56). orientation(fcgt_3,159,56). coords(fcgt_3,103,163,56).  visible(fcgt_3,56). happensAt(inactive(fcgt_3),56).
orientation(fcgt_0,143,57). coords(fcgt_0,195,148,57).  visible(fcgt_0,57). happensAt(walking(fcgt_0),57). orientation(fcgt_3,159,57). coords(fcgt_3,103,163,57).  visible(fcgt_3,57). happensAt(inactive(fcgt_3),57).
orientation(fcgt_0,143,58). coords(fcgt_0,197,149,58).  visible(fcgt_0,58). happensAt(walking(fcgt_0),58). orientation(fcgt_3,159,58). coords(fcgt_3,103,163,58).  visible(fcgt_3,58). happensAt(inactive(fcgt_3),58).
orientation(fcgt_0,143,59). coords(fcgt_0,197,149,59).  visible(fcgt_0,59). happensAt(walking(fcgt_0),59). orientation(fcgt_3,159,59). coords(fcgt_3,103,163,59).  visible(fcgt_3,59). happensAt(inactive(fcgt_3),59).
orientation(fcgt_0,143,60). coords(fcgt_0,198,150,60).  visible(fcgt_0,60). happensAt(walking(fcgt_0),60). orientation(fcgt_3,159,60). coords(fcgt_3,103,163,60).  visible(fcgt_3,60). happensAt(inactive(fcgt_3),60).
orientation(fcgt_0,143,61). coords(fcgt_0,198,150,61).  visible(fcgt_0,61). happensAt(walking(fcgt_0),61). orientation(fcgt_3,159,61). coords(fcgt_3,103,163,61).  visible(fcgt_3,61). happensAt(inactive(fcgt_3),61).
orientation(fcgt_0,143,62). coords(fcgt_0,199,151,62).  visible(fcgt_0,62). happensAt(walking(fcgt_0),62). orientation(fcgt_3,159,62). coords(fcgt_3,103,163,62).  visible(fcgt_3,62). happensAt(inactive(fcgt_3),62).
orientation(fcgt_0,143,63). coords(fcgt_0,199,151,63).  visible(fcgt_0,63). happensAt(walking(fcgt_0),63). orientation(fcgt_3,159,63). coords(fcgt_3,103,163,63).  visible(fcgt_3,63). happensAt(inactive(fcgt_3),63).
orientation(fcgt_0,143,64). coords(fcgt_0,199,151,64).  visible(fcgt_0,64). happensAt(walking(fcgt_0),64). orientation(fcgt_3,159,64). coords(fcgt_3,103,163,64).  visible(fcgt_3,64). happensAt(inactive(fcgt_3),64).
orientation(fcgt_0,143,65). coords(fcgt_0,199,151,65).  visible(fcgt_0,65). happensAt(inactive(fcgt_0),65). orientation(fcgt_3,159,65). coords(fcgt_3,103,163,65).  visible(fcgt_3,65). happensAt(inactive(fcgt_3),65).
orientation(fcgt_0,143,66). coords(fcgt_0,199,151,66).  visible(fcgt_0,66). happensAt(inactive(fcgt_0),66). orientation(fcgt_3,159,66). coords(fcgt_3,103,163,66).  visible(fcgt_3,66). happensAt(inactive(fcgt_3),66).
orientation(fcgt_0,143,67). coords(fcgt_0,199,151,67).  visible(fcgt_0,67). happensAt(inactive(fcgt_0),67). orientation(fcgt_3,159,67). coords(fcgt_3,103,163,67).  visible(fcgt_3,67). happensAt(inactive(fcgt_3),67).
orientation(fcgt_0,143,68). coords(fcgt_0,199,151,68).  visible(fcgt_0,68). happensAt(inactive(fcgt_0),68). orientation(fcgt_3,159,68). coords(fcgt_3,103,163,68).  visible(fcgt_3,68). happensAt(inactive(fcgt_3),68).
orientation(fcgt_0,143,69). coords(fcgt_0,199,151,69).  visible(fcgt_0,69). happensAt(inactive(fcgt_0),69). orientation(fcgt_3,159,69). coords(fcgt_3,103,163,69).  visible(fcgt_3,69). happensAt(inactive(fcgt_3),69).
orientation(fcgt_0,143,70). coords(fcgt_0,199,151,70).  visible(fcgt_0,70). happensAt(inactive(fcgt_0),70). orientation(fcgt_3,159,70). coords(fcgt_3,103,163,70).  visible(fcgt_3,70). happensAt(inactive(fcgt_3),70).
orientation(fcgt_0,143,71). coords(fcgt_0,199,151,71).  visible(fcgt_0,71). happensAt(inactive(fcgt_0),71). orientation(fcgt_3,159,71). coords(fcgt_3,103,163,71).  visible(fcgt_3,71). happensAt(inactive(fcgt_3),71).
orientation(fcgt_0,143,72). coords(fcgt_0,199,151,72).  visible(fcgt_0,72). happensAt(inactive(fcgt_0),72). orientation(fcgt_3,159,72). coords(fcgt_3,103,163,72).  visible(fcgt_3,72). happensAt(inactive(fcgt_3),72).
#show holdsAt/2.