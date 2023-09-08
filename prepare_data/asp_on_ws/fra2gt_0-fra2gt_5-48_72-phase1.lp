
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

	orientation(fra2gt_0,150,48). coords(fra2gt_0,158,129,48).  visible(fra2gt_0,48). happensAt(active(fra2gt_0),48). orientation(fra2gt_5,150,48). coords(fra2gt_5,39,106,48).  visible(fra2gt_5,48). happensAt(walking(fra2gt_5),48).
orientation(fra2gt_0,150,49). coords(fra2gt_0,158,129,49).  visible(fra2gt_0,49). happensAt(active(fra2gt_0),49). orientation(fra2gt_5,150,49). coords(fra2gt_5,39,106,49).  visible(fra2gt_5,49). happensAt(walking(fra2gt_5),49).
orientation(fra2gt_0,150,50). coords(fra2gt_0,158,129,50).  visible(fra2gt_0,50). happensAt(active(fra2gt_0),50). orientation(fra2gt_5,150,50). coords(fra2gt_5,39,106,50).  visible(fra2gt_5,50). happensAt(walking(fra2gt_5),50).
orientation(fra2gt_0,150,51). coords(fra2gt_0,158,129,51).  visible(fra2gt_0,51). happensAt(active(fra2gt_0),51). orientation(fra2gt_5,150,51). coords(fra2gt_5,38,107,51).  visible(fra2gt_5,51). happensAt(walking(fra2gt_5),51).
orientation(fra2gt_0,150,52). coords(fra2gt_0,157,129,52).  visible(fra2gt_0,52). happensAt(active(fra2gt_0),52). orientation(fra2gt_5,150,52). coords(fra2gt_5,38,107,52).  visible(fra2gt_5,52). happensAt(walking(fra2gt_5),52).
orientation(fra2gt_0,150,53). coords(fra2gt_0,157,129,53).  visible(fra2gt_0,53). happensAt(active(fra2gt_0),53). orientation(fra2gt_5,150,53). coords(fra2gt_5,38,107,53).  visible(fra2gt_5,53). happensAt(walking(fra2gt_5),53).
orientation(fra2gt_0,150,54). coords(fra2gt_0,157,129,54).  visible(fra2gt_0,54). happensAt(inactive(fra2gt_0),54). orientation(fra2gt_5,150,54). coords(fra2gt_5,38,107,54).  visible(fra2gt_5,54). happensAt(walking(fra2gt_5),54).
orientation(fra2gt_0,150,55). coords(fra2gt_0,157,129,55).  visible(fra2gt_0,55). happensAt(inactive(fra2gt_0),55). orientation(fra2gt_5,150,55). coords(fra2gt_5,37,108,55).  visible(fra2gt_5,55). happensAt(walking(fra2gt_5),55).
orientation(fra2gt_0,150,56). coords(fra2gt_0,157,129,56).  visible(fra2gt_0,56). happensAt(inactive(fra2gt_0),56). orientation(fra2gt_5,150,56). coords(fra2gt_5,37,109,56).  visible(fra2gt_5,56). happensAt(walking(fra2gt_5),56).
orientation(fra2gt_0,150,57). coords(fra2gt_0,157,129,57).  visible(fra2gt_0,57). happensAt(inactive(fra2gt_0),57). orientation(fra2gt_5,150,57). coords(fra2gt_5,37,109,57).  visible(fra2gt_5,57). happensAt(walking(fra2gt_5),57).
orientation(fra2gt_0,150,58). coords(fra2gt_0,157,129,58).  visible(fra2gt_0,58). happensAt(inactive(fra2gt_0),58). orientation(fra2gt_5,150,58). coords(fra2gt_5,36,110,58).  visible(fra2gt_5,58). happensAt(walking(fra2gt_5),58).
orientation(fra2gt_0,150,59). coords(fra2gt_0,157,129,59).  visible(fra2gt_0,59). happensAt(inactive(fra2gt_0),59). orientation(fra2gt_5,150,59). coords(fra2gt_5,35,110,59).  visible(fra2gt_5,59). happensAt(walking(fra2gt_5),59).
orientation(fra2gt_0,150,60). coords(fra2gt_0,157,129,60).  visible(fra2gt_0,60). happensAt(inactive(fra2gt_0),60). orientation(fra2gt_5,150,60). coords(fra2gt_5,35,110,60).  visible(fra2gt_5,60). happensAt(walking(fra2gt_5),60).
orientation(fra2gt_0,150,61). coords(fra2gt_0,157,129,61).  visible(fra2gt_0,61). happensAt(inactive(fra2gt_0),61). orientation(fra2gt_5,150,61). coords(fra2gt_5,35,110,61).  visible(fra2gt_5,61). happensAt(walking(fra2gt_5),61).
orientation(fra2gt_0,150,62). coords(fra2gt_0,157,129,62).  visible(fra2gt_0,62). happensAt(inactive(fra2gt_0),62). orientation(fra2gt_5,150,62). coords(fra2gt_5,35,110,62).  visible(fra2gt_5,62). happensAt(walking(fra2gt_5),62).
orientation(fra2gt_0,150,63). coords(fra2gt_0,157,129,63).  visible(fra2gt_0,63). happensAt(inactive(fra2gt_0),63). orientation(fra2gt_5,150,63). coords(fra2gt_5,35,110,63).  visible(fra2gt_5,63). happensAt(walking(fra2gt_5),63).
orientation(fra2gt_0,150,64). coords(fra2gt_0,157,129,64).  visible(fra2gt_0,64). happensAt(inactive(fra2gt_0),64). orientation(fra2gt_5,150,64). coords(fra2gt_5,35,110,64).  visible(fra2gt_5,64). happensAt(walking(fra2gt_5),64).
orientation(fra2gt_0,150,65). coords(fra2gt_0,157,129,65).  visible(fra2gt_0,65). happensAt(inactive(fra2gt_0),65). orientation(fra2gt_5,150,65). coords(fra2gt_5,34,110,65).  visible(fra2gt_5,65). happensAt(walking(fra2gt_5),65).
orientation(fra2gt_0,150,66). coords(fra2gt_0,157,129,66).  visible(fra2gt_0,66). happensAt(inactive(fra2gt_0),66). orientation(fra2gt_5,150,66). coords(fra2gt_5,34,110,66).  visible(fra2gt_5,66). happensAt(walking(fra2gt_5),66).
orientation(fra2gt_0,150,67). coords(fra2gt_0,157,129,67).  visible(fra2gt_0,67). happensAt(inactive(fra2gt_0),67). orientation(fra2gt_5,150,67). coords(fra2gt_5,34,110,67).  visible(fra2gt_5,67). happensAt(walking(fra2gt_5),67).
orientation(fra2gt_0,150,68). coords(fra2gt_0,157,129,68).  visible(fra2gt_0,68). happensAt(inactive(fra2gt_0),68). orientation(fra2gt_5,150,68). coords(fra2gt_5,34,110,68).  visible(fra2gt_5,68). happensAt(walking(fra2gt_5),68).
orientation(fra2gt_0,150,69). coords(fra2gt_0,157,129,69).  visible(fra2gt_0,69). happensAt(active(fra2gt_0),69). orientation(fra2gt_5,150,69). coords(fra2gt_5,32,110,69).  visible(fra2gt_5,69). happensAt(walking(fra2gt_5),69).
orientation(fra2gt_0,150,70). coords(fra2gt_0,157,129,70).  visible(fra2gt_0,70). happensAt(active(fra2gt_0),70). orientation(fra2gt_5,150,70). coords(fra2gt_5,32,110,70).  visible(fra2gt_5,70). happensAt(walking(fra2gt_5),70).
orientation(fra2gt_0,150,71). coords(fra2gt_0,157,129,71).  visible(fra2gt_0,71). happensAt(active(fra2gt_0),71). orientation(fra2gt_5,150,71). coords(fra2gt_5,32,110,71).  visible(fra2gt_5,71). happensAt(walking(fra2gt_5),71).
orientation(fra2gt_0,150,72). coords(fra2gt_0,157,129,72).  visible(fra2gt_0,72). happensAt(active(fra2gt_0),72). orientation(fra2gt_5,150,72). coords(fra2gt_5,32,112,72).  visible(fra2gt_5,72). happensAt(walking(fra2gt_5),72).
#show holdsAt/2.