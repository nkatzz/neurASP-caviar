
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

	orientation(wk2gt_1,143,48). coords(wk2gt_1,109,56,48).  visible(wk2gt_1,48). happensAt(walking(wk2gt_1),48). orientation(wk2gt_2,141,48). coords(wk2gt_2,98,63,48).  visible(wk2gt_2,48). happensAt(walking(wk2gt_2),48).
orientation(wk2gt_1,143,49). coords(wk2gt_1,110,56,49).  visible(wk2gt_1,49). happensAt(walking(wk2gt_1),49). orientation(wk2gt_2,141,49). coords(wk2gt_2,98,63,49).  visible(wk2gt_2,49). happensAt(walking(wk2gt_2),49).
orientation(wk2gt_1,143,50). coords(wk2gt_1,110,56,50).  visible(wk2gt_1,50). happensAt(walking(wk2gt_1),50). orientation(wk2gt_2,141,50). coords(wk2gt_2,98,63,50).  visible(wk2gt_2,50). happensAt(walking(wk2gt_2),50).
orientation(wk2gt_1,143,51). coords(wk2gt_1,110,56,51).  visible(wk2gt_1,51). happensAt(walking(wk2gt_1),51). orientation(wk2gt_2,141,51). coords(wk2gt_2,98,63,51).  visible(wk2gt_2,51). happensAt(walking(wk2gt_2),51).
orientation(wk2gt_1,143,52). coords(wk2gt_1,110,56,52).  visible(wk2gt_1,52). happensAt(walking(wk2gt_1),52). orientation(wk2gt_2,141,52). coords(wk2gt_2,98,63,52).  visible(wk2gt_2,52). happensAt(walking(wk2gt_2),52).
orientation(wk2gt_1,143,53). coords(wk2gt_1,110,56,53).  visible(wk2gt_1,53). happensAt(walking(wk2gt_1),53). orientation(wk2gt_2,141,53). coords(wk2gt_2,98,63,53).  visible(wk2gt_2,53). happensAt(walking(wk2gt_2),53).
orientation(wk2gt_1,143,54). coords(wk2gt_1,110,56,54).  visible(wk2gt_1,54). happensAt(walking(wk2gt_1),54). orientation(wk2gt_2,141,54). coords(wk2gt_2,98,63,54).  visible(wk2gt_2,54). happensAt(walking(wk2gt_2),54).
orientation(wk2gt_1,143,55). coords(wk2gt_1,110,56,55).  visible(wk2gt_1,55). happensAt(walking(wk2gt_1),55). orientation(wk2gt_2,141,55). coords(wk2gt_2,98,63,55).  visible(wk2gt_2,55). happensAt(walking(wk2gt_2),55).
orientation(wk2gt_1,143,56). coords(wk2gt_1,110,56,56).  visible(wk2gt_1,56). happensAt(walking(wk2gt_1),56). orientation(wk2gt_2,141,56). coords(wk2gt_2,98,63,56).  visible(wk2gt_2,56). happensAt(walking(wk2gt_2),56).
orientation(wk2gt_1,143,57). coords(wk2gt_1,111,56,57).  visible(wk2gt_1,57). happensAt(walking(wk2gt_1),57). orientation(wk2gt_2,141,57). coords(wk2gt_2,98,63,57).  visible(wk2gt_2,57). happensAt(walking(wk2gt_2),57).
orientation(wk2gt_1,143,58). coords(wk2gt_1,112,56,58).  visible(wk2gt_1,58). happensAt(walking(wk2gt_1),58). orientation(wk2gt_2,141,58). coords(wk2gt_2,98,63,58).  visible(wk2gt_2,58). happensAt(walking(wk2gt_2),58).
orientation(wk2gt_1,143,59). coords(wk2gt_1,112,55,59).  visible(wk2gt_1,59). happensAt(walking(wk2gt_1),59). orientation(wk2gt_2,141,59). coords(wk2gt_2,98,63,59).  visible(wk2gt_2,59). happensAt(walking(wk2gt_2),59).
orientation(wk2gt_1,140,60). coords(wk2gt_1,112,55,60).  visible(wk2gt_1,60). happensAt(walking(wk2gt_1),60). orientation(wk2gt_2,141,60). coords(wk2gt_2,98,63,60).  visible(wk2gt_2,60). happensAt(walking(wk2gt_2),60).
orientation(wk2gt_1,140,61). coords(wk2gt_1,112,54,61).  visible(wk2gt_1,61). happensAt(walking(wk2gt_1),61). orientation(wk2gt_2,141,61). coords(wk2gt_2,98,63,61).  visible(wk2gt_2,61). happensAt(walking(wk2gt_2),61).
orientation(wk2gt_1,139,62). coords(wk2gt_1,112,53,62).  visible(wk2gt_1,62). happensAt(walking(wk2gt_1),62). orientation(wk2gt_2,141,62). coords(wk2gt_2,98,63,62).  visible(wk2gt_2,62). happensAt(walking(wk2gt_2),62).
orientation(wk2gt_1,140,63). coords(wk2gt_1,112,53,63).  visible(wk2gt_1,63). happensAt(walking(wk2gt_1),63). orientation(wk2gt_2,141,63). coords(wk2gt_2,98,63,63).  visible(wk2gt_2,63). happensAt(walking(wk2gt_2),63).
orientation(wk2gt_1,134,64). coords(wk2gt_1,112,53,64).  visible(wk2gt_1,64). happensAt(active(wk2gt_1),64). orientation(wk2gt_2,141,64). coords(wk2gt_2,98,63,64).  visible(wk2gt_2,64). happensAt(active(wk2gt_2),64).
orientation(wk2gt_1,135,65). coords(wk2gt_1,112,53,65).  visible(wk2gt_1,65). happensAt(active(wk2gt_1),65). orientation(wk2gt_2,141,65). coords(wk2gt_2,98,63,65).  visible(wk2gt_2,65). happensAt(active(wk2gt_2),65).
orientation(wk2gt_1,134,66). coords(wk2gt_1,113,53,66).  visible(wk2gt_1,66). happensAt(active(wk2gt_1),66). orientation(wk2gt_2,141,66). coords(wk2gt_2,98,63,66).  visible(wk2gt_2,66). happensAt(active(wk2gt_2),66).
orientation(wk2gt_1,142,67). coords(wk2gt_1,113,53,67).  visible(wk2gt_1,67). happensAt(active(wk2gt_1),67). orientation(wk2gt_2,141,67). coords(wk2gt_2,98,63,67).  visible(wk2gt_2,67). happensAt(active(wk2gt_2),67).
orientation(wk2gt_1,135,68). coords(wk2gt_1,113,53,68).  visible(wk2gt_1,68). happensAt(active(wk2gt_1),68). orientation(wk2gt_2,141,68). coords(wk2gt_2,98,63,68).  visible(wk2gt_2,68). happensAt(active(wk2gt_2),68).
orientation(wk2gt_1,135,69). coords(wk2gt_1,113,53,69).  visible(wk2gt_1,69). happensAt(active(wk2gt_1),69). orientation(wk2gt_2,141,69). coords(wk2gt_2,98,63,69).  visible(wk2gt_2,69). happensAt(active(wk2gt_2),69).
orientation(wk2gt_1,135,70). coords(wk2gt_1,113,53,70).  visible(wk2gt_1,70). happensAt(active(wk2gt_1),70). orientation(wk2gt_2,141,70). coords(wk2gt_2,98,63,70).  visible(wk2gt_2,70). happensAt(active(wk2gt_2),70).
orientation(wk2gt_1,135,71). coords(wk2gt_1,114,53,71).  visible(wk2gt_1,71). happensAt(active(wk2gt_1),71). orientation(wk2gt_2,141,71). coords(wk2gt_2,98,63,71).  visible(wk2gt_2,71). happensAt(active(wk2gt_2),71).
orientation(wk2gt_1,135,72). coords(wk2gt_1,114,53,72).  visible(wk2gt_1,72). happensAt(active(wk2gt_1),72). orientation(wk2gt_2,141,72). coords(wk2gt_2,98,63,72).  visible(wk2gt_2,72). happensAt(active(wk2gt_2),72).
#show holdsAt/2.