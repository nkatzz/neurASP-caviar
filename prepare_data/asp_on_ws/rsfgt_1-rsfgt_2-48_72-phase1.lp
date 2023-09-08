
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

	orientation(rsfgt_1,150,48). coords(rsfgt_1,183,134,48).  visible(rsfgt_1,48). happensAt(walking(rsfgt_1),48). orientation(rsfgt_2,170,48). coords(rsfgt_2,212,247,48).  visible(rsfgt_2,48). happensAt(walking(rsfgt_2),48).
orientation(rsfgt_1,150,49). coords(rsfgt_1,183,133,49).  visible(rsfgt_1,49). happensAt(walking(rsfgt_1),49). orientation(rsfgt_2,170,49). coords(rsfgt_2,212,251,49).  visible(rsfgt_2,49). happensAt(walking(rsfgt_2),49).
orientation(rsfgt_1,150,50). coords(rsfgt_1,184,133,50).  visible(rsfgt_1,50). happensAt(walking(rsfgt_1),50). orientation(rsfgt_2,170,50). coords(rsfgt_2,214,252,50).  visible(rsfgt_2,50). happensAt(walking(rsfgt_2),50).
orientation(rsfgt_1,150,51). coords(rsfgt_1,184,131,51).  visible(rsfgt_1,51). happensAt(walking(rsfgt_1),51). orientation(rsfgt_2,170,51). coords(rsfgt_2,216,254,51).  visible(rsfgt_2,51). happensAt(walking(rsfgt_2),51).
orientation(rsfgt_1,150,52). coords(rsfgt_1,183,128,52).  visible(rsfgt_1,52). happensAt(walking(rsfgt_1),52). orientation(rsfgt_2,170,52). coords(rsfgt_2,216,255,52).  visible(rsfgt_2,52). happensAt(walking(rsfgt_2),52).
orientation(rsfgt_1,150,53). coords(rsfgt_1,183,127,53).  visible(rsfgt_1,53). happensAt(walking(rsfgt_1),53). orientation(rsfgt_2,170,53). coords(rsfgt_2,216,255,53).  visible(rsfgt_2,53). happensAt(walking(rsfgt_2),53).
orientation(rsfgt_1,150,54). coords(rsfgt_1,182,127,54).  visible(rsfgt_1,54). happensAt(walking(rsfgt_1),54). orientation(rsfgt_2,170,54). coords(rsfgt_2,216,255,54).  visible(rsfgt_2,54). happensAt(walking(rsfgt_2),54).
orientation(rsfgt_1,150,55). coords(rsfgt_1,179,126,55).  visible(rsfgt_1,55). happensAt(walking(rsfgt_1),55). orientation(rsfgt_2,170,55). coords(rsfgt_2,216,258,55).  visible(rsfgt_2,55). happensAt(walking(rsfgt_2),55).
orientation(rsfgt_1,150,56). coords(rsfgt_1,178,126,56).  visible(rsfgt_1,56). happensAt(walking(rsfgt_1),56). orientation(rsfgt_2,170,56). coords(rsfgt_2,218,261,56).  visible(rsfgt_2,56). happensAt(walking(rsfgt_2),56).
orientation(rsfgt_1,150,57). coords(rsfgt_1,178,123,57).  visible(rsfgt_1,57). happensAt(walking(rsfgt_1),57). orientation(rsfgt_2,170,57). coords(rsfgt_2,219,263,57).  visible(rsfgt_2,57). happensAt(walking(rsfgt_2),57).
orientation(rsfgt_1,150,58). coords(rsfgt_1,178,122,58).  visible(rsfgt_1,58). happensAt(walking(rsfgt_1),58). orientation(rsfgt_2,170,58). coords(rsfgt_2,219,266,58).  visible(rsfgt_2,58). happensAt(walking(rsfgt_2),58).
orientation(rsfgt_1,150,59). coords(rsfgt_1,177,121,59).  visible(rsfgt_1,59). happensAt(walking(rsfgt_1),59). orientation(rsfgt_2,170,59). coords(rsfgt_2,220,269,59).  visible(rsfgt_2,59). happensAt(walking(rsfgt_2),59).
orientation(rsfgt_1,150,60). coords(rsfgt_1,177,117,60).  visible(rsfgt_1,60). happensAt(walking(rsfgt_1),60). orientation(rsfgt_2,170,60). coords(rsfgt_2,221,271,60).  visible(rsfgt_2,60). happensAt(walking(rsfgt_2),60).
orientation(rsfgt_1,150,61). coords(rsfgt_1,176,117,61).  visible(rsfgt_1,61). happensAt(walking(rsfgt_1),61). orientation(rsfgt_2,170,61). coords(rsfgt_2,222,273,61).  visible(rsfgt_2,61). happensAt(walking(rsfgt_2),61).
orientation(rsfgt_1,150,62). coords(rsfgt_1,175,117,62).  visible(rsfgt_1,62). happensAt(walking(rsfgt_1),62). orientation(rsfgt_2,170,62). coords(rsfgt_2,222,274,62).  visible(rsfgt_2,62). happensAt(walking(rsfgt_2),62).
orientation(rsfgt_1,150,63). coords(rsfgt_1,174,117,63).  visible(rsfgt_1,63). happensAt(walking(rsfgt_1),63). orientation(rsfgt_2,170,63). coords(rsfgt_2,224,275,63).  visible(rsfgt_2,63). happensAt(walking(rsfgt_2),63).
orientation(rsfgt_1,150,64). coords(rsfgt_1,174,117,64).  visible(rsfgt_1,64). happensAt(walking(rsfgt_1),64). orientation(rsfgt_2,170,64). coords(rsfgt_2,224,273,64).  visible(rsfgt_2,64). happensAt(walking(rsfgt_2),64).
orientation(rsfgt_1,150,65). coords(rsfgt_1,174,117,65).  visible(rsfgt_1,65). happensAt(walking(rsfgt_1),65). orientation(rsfgt_2,25,65). coords(rsfgt_2,225,274,65).  visible(rsfgt_2,65). happensAt(walking(rsfgt_2),65).
orientation(rsfgt_1,150,66). coords(rsfgt_1,174,117,66).  visible(rsfgt_1,66). happensAt(walking(rsfgt_1),66). orientation(rsfgt_2,25,66). coords(rsfgt_2,226,275,66).  visible(rsfgt_2,66). happensAt(walking(rsfgt_2),66).
orientation(rsfgt_1,150,67). coords(rsfgt_1,174,117,67).  visible(rsfgt_1,67). happensAt(walking(rsfgt_1),67). orientation(rsfgt_2,25,67). coords(rsfgt_2,227,275,67).  visible(rsfgt_2,67). happensAt(walking(rsfgt_2),67).
orientation(rsfgt_1,150,68). coords(rsfgt_1,173,117,68).  visible(rsfgt_1,68). happensAt(walking(rsfgt_1),68). orientation(rsfgt_2,25,68). coords(rsfgt_2,228,276,68).  visible(rsfgt_2,68). happensAt(walking(rsfgt_2),68).
orientation(rsfgt_1,150,69). coords(rsfgt_1,173,118,69).  visible(rsfgt_1,69). happensAt(walking(rsfgt_1),69). orientation(rsfgt_2,25,69). coords(rsfgt_2,229,277,69).  visible(rsfgt_2,69). happensAt(walking(rsfgt_2),69).
orientation(rsfgt_1,150,70). coords(rsfgt_1,172,117,70).  visible(rsfgt_1,70). happensAt(walking(rsfgt_1),70). orientation(rsfgt_2,25,70). coords(rsfgt_2,230,278,70).  visible(rsfgt_2,70). happensAt(walking(rsfgt_2),70).
orientation(rsfgt_1,150,71). coords(rsfgt_1,170,116,71).  visible(rsfgt_1,71). happensAt(walking(rsfgt_1),71). orientation(rsfgt_2,25,71). coords(rsfgt_2,232,279,71).  visible(rsfgt_2,71). happensAt(walking(rsfgt_2),71).
orientation(rsfgt_1,150,72). coords(rsfgt_1,170,116,72).  visible(rsfgt_1,72). happensAt(walking(rsfgt_1),72). orientation(rsfgt_2,25,72). coords(rsfgt_2,233,281,72).  visible(rsfgt_2,72). happensAt(walking(rsfgt_2),72).
#show holdsAt/2.