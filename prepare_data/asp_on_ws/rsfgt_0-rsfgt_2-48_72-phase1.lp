
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

	orientation(rsfgt_0,160,48). coords(rsfgt_0,90,134,48).  visible(rsfgt_0,48). happensAt(walking(rsfgt_0),48). orientation(rsfgt_2,170,48). coords(rsfgt_2,212,247,48).  visible(rsfgt_2,48). happensAt(walking(rsfgt_2),48).
orientation(rsfgt_0,160,49). coords(rsfgt_0,90,134,49).  visible(rsfgt_0,49). happensAt(walking(rsfgt_0),49). orientation(rsfgt_2,170,49). coords(rsfgt_2,212,251,49).  visible(rsfgt_2,49). happensAt(walking(rsfgt_2),49).
orientation(rsfgt_0,160,50). coords(rsfgt_0,89,134,50).  visible(rsfgt_0,50). happensAt(walking(rsfgt_0),50). orientation(rsfgt_2,170,50). coords(rsfgt_2,214,252,50).  visible(rsfgt_2,50). happensAt(walking(rsfgt_2),50).
orientation(rsfgt_0,160,51). coords(rsfgt_0,87,134,51).  visible(rsfgt_0,51). happensAt(walking(rsfgt_0),51). orientation(rsfgt_2,170,51). coords(rsfgt_2,216,254,51).  visible(rsfgt_2,51). happensAt(walking(rsfgt_2),51).
orientation(rsfgt_0,160,52). coords(rsfgt_0,85,134,52).  visible(rsfgt_0,52). happensAt(walking(rsfgt_0),52). orientation(rsfgt_2,170,52). coords(rsfgt_2,216,255,52).  visible(rsfgt_2,52). happensAt(walking(rsfgt_2),52).
orientation(rsfgt_0,160,53). coords(rsfgt_0,82,134,53).  visible(rsfgt_0,53). happensAt(walking(rsfgt_0),53). orientation(rsfgt_2,170,53). coords(rsfgt_2,216,255,53).  visible(rsfgt_2,53). happensAt(walking(rsfgt_2),53).
orientation(rsfgt_0,160,54). coords(rsfgt_0,82,134,54).  visible(rsfgt_0,54). happensAt(walking(rsfgt_0),54). orientation(rsfgt_2,170,54). coords(rsfgt_2,216,255,54).  visible(rsfgt_2,54). happensAt(walking(rsfgt_2),54).
orientation(rsfgt_0,160,55). coords(rsfgt_0,81,134,55).  visible(rsfgt_0,55). happensAt(walking(rsfgt_0),55). orientation(rsfgt_2,170,55). coords(rsfgt_2,216,258,55).  visible(rsfgt_2,55). happensAt(walking(rsfgt_2),55).
orientation(rsfgt_0,160,56). coords(rsfgt_0,80,134,56).  visible(rsfgt_0,56). happensAt(walking(rsfgt_0),56). orientation(rsfgt_2,170,56). coords(rsfgt_2,218,261,56).  visible(rsfgt_2,56). happensAt(walking(rsfgt_2),56).
orientation(rsfgt_0,160,57). coords(rsfgt_0,80,134,57).  visible(rsfgt_0,57). happensAt(walking(rsfgt_0),57). orientation(rsfgt_2,170,57). coords(rsfgt_2,219,263,57).  visible(rsfgt_2,57). happensAt(walking(rsfgt_2),57).
orientation(rsfgt_0,160,58). coords(rsfgt_0,80,134,58).  visible(rsfgt_0,58). happensAt(walking(rsfgt_0),58). orientation(rsfgt_2,170,58). coords(rsfgt_2,219,266,58).  visible(rsfgt_2,58). happensAt(walking(rsfgt_2),58).
orientation(rsfgt_0,160,59). coords(rsfgt_0,80,136,59).  visible(rsfgt_0,59). happensAt(walking(rsfgt_0),59). orientation(rsfgt_2,170,59). coords(rsfgt_2,220,269,59).  visible(rsfgt_2,59). happensAt(walking(rsfgt_2),59).
orientation(rsfgt_0,160,60). coords(rsfgt_0,79,136,60).  visible(rsfgt_0,60). happensAt(walking(rsfgt_0),60). orientation(rsfgt_2,170,60). coords(rsfgt_2,221,271,60).  visible(rsfgt_2,60). happensAt(walking(rsfgt_2),60).
orientation(rsfgt_0,160,61). coords(rsfgt_0,78,137,61).  visible(rsfgt_0,61). happensAt(walking(rsfgt_0),61). orientation(rsfgt_2,170,61). coords(rsfgt_2,222,273,61).  visible(rsfgt_2,61). happensAt(walking(rsfgt_2),61).
orientation(rsfgt_0,160,62). coords(rsfgt_0,78,137,62).  visible(rsfgt_0,62). happensAt(walking(rsfgt_0),62). orientation(rsfgt_2,170,62). coords(rsfgt_2,222,274,62).  visible(rsfgt_2,62). happensAt(walking(rsfgt_2),62).
orientation(rsfgt_0,160,63). coords(rsfgt_0,78,137,63).  visible(rsfgt_0,63). happensAt(walking(rsfgt_0),63). orientation(rsfgt_2,170,63). coords(rsfgt_2,224,275,63).  visible(rsfgt_2,63). happensAt(walking(rsfgt_2),63).
orientation(rsfgt_0,160,64). coords(rsfgt_0,75,136,64).  visible(rsfgt_0,64). happensAt(walking(rsfgt_0),64). orientation(rsfgt_2,170,64). coords(rsfgt_2,224,273,64).  visible(rsfgt_2,64). happensAt(walking(rsfgt_2),64).
orientation(rsfgt_0,160,65). coords(rsfgt_0,73,137,65).  visible(rsfgt_0,65). happensAt(walking(rsfgt_0),65). orientation(rsfgt_2,25,65). coords(rsfgt_2,225,274,65).  visible(rsfgt_2,65). happensAt(walking(rsfgt_2),65).
orientation(rsfgt_0,160,66). coords(rsfgt_0,72,138,66).  visible(rsfgt_0,66). happensAt(walking(rsfgt_0),66). orientation(rsfgt_2,25,66). coords(rsfgt_2,226,275,66).  visible(rsfgt_2,66). happensAt(walking(rsfgt_2),66).
orientation(rsfgt_0,160,67). coords(rsfgt_0,71,139,67).  visible(rsfgt_0,67). happensAt(walking(rsfgt_0),67). orientation(rsfgt_2,25,67). coords(rsfgt_2,227,275,67).  visible(rsfgt_2,67). happensAt(walking(rsfgt_2),67).
orientation(rsfgt_0,160,68). coords(rsfgt_0,71,139,68).  visible(rsfgt_0,68). happensAt(walking(rsfgt_0),68). orientation(rsfgt_2,25,68). coords(rsfgt_2,228,276,68).  visible(rsfgt_2,68). happensAt(walking(rsfgt_2),68).
orientation(rsfgt_0,160,69). coords(rsfgt_0,71,139,69).  visible(rsfgt_0,69). happensAt(walking(rsfgt_0),69). orientation(rsfgt_2,25,69). coords(rsfgt_2,229,277,69).  visible(rsfgt_2,69). happensAt(walking(rsfgt_2),69).
orientation(rsfgt_0,160,70). coords(rsfgt_0,71,139,70).  visible(rsfgt_0,70). happensAt(walking(rsfgt_0),70). orientation(rsfgt_2,25,70). coords(rsfgt_2,230,278,70).  visible(rsfgt_2,70). happensAt(walking(rsfgt_2),70).
orientation(rsfgt_0,160,71). coords(rsfgt_0,71,140,71).  visible(rsfgt_0,71). happensAt(walking(rsfgt_0),71). orientation(rsfgt_2,25,71). coords(rsfgt_2,232,279,71).  visible(rsfgt_2,71). happensAt(walking(rsfgt_2),71).
orientation(rsfgt_0,160,72). coords(rsfgt_0,70,141,72).  visible(rsfgt_0,72). happensAt(walking(rsfgt_0),72). orientation(rsfgt_2,25,72). coords(rsfgt_2,233,281,72).  visible(rsfgt_2,72). happensAt(walking(rsfgt_2),72).
#show holdsAt/2.