
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

	orientation(lb1gt_0,148,48). coords(lb1gt_0,182,119,48).  visible(lb1gt_0,48). happensAt(walking(lb1gt_0),48). orientation(lb1gt_1,147,48). coords(lb1gt_1,73,92,48).  visible(lb1gt_1,48). happensAt(walking(lb1gt_1),48).
orientation(lb1gt_0,148,49). coords(lb1gt_0,182,119,49).  visible(lb1gt_0,49). happensAt(walking(lb1gt_0),49). orientation(lb1gt_1,147,49). coords(lb1gt_1,73,92,49).  visible(lb1gt_1,49). happensAt(walking(lb1gt_1),49).
orientation(lb1gt_0,148,50). coords(lb1gt_0,182,119,50).  visible(lb1gt_0,50). happensAt(walking(lb1gt_0),50). orientation(lb1gt_1,147,50). coords(lb1gt_1,73,92,50).  visible(lb1gt_1,50). happensAt(walking(lb1gt_1),50).
orientation(lb1gt_0,148,51). coords(lb1gt_0,182,119,51).  visible(lb1gt_0,51). happensAt(walking(lb1gt_0),51). orientation(lb1gt_1,147,51). coords(lb1gt_1,73,93,51).  visible(lb1gt_1,51). happensAt(walking(lb1gt_1),51).
orientation(lb1gt_0,148,52). coords(lb1gt_0,182,119,52).  visible(lb1gt_0,52). happensAt(walking(lb1gt_0),52). orientation(lb1gt_1,147,52). coords(lb1gt_1,73,94,52).  visible(lb1gt_1,52). happensAt(walking(lb1gt_1),52).
orientation(lb1gt_0,148,53). coords(lb1gt_0,182,119,53).  visible(lb1gt_0,53). happensAt(walking(lb1gt_0),53). orientation(lb1gt_1,147,53). coords(lb1gt_1,73,95,53).  visible(lb1gt_1,53). happensAt(walking(lb1gt_1),53).
orientation(lb1gt_0,148,54). coords(lb1gt_0,182,119,54).  visible(lb1gt_0,54). happensAt(walking(lb1gt_0),54). orientation(lb1gt_1,147,54). coords(lb1gt_1,73,97,54).  visible(lb1gt_1,54). happensAt(walking(lb1gt_1),54).
orientation(lb1gt_0,148,55). coords(lb1gt_0,182,119,55).  visible(lb1gt_0,55). happensAt(walking(lb1gt_0),55). orientation(lb1gt_1,147,55). coords(lb1gt_1,73,98,55).  visible(lb1gt_1,55). happensAt(walking(lb1gt_1),55).
orientation(lb1gt_0,148,56). coords(lb1gt_0,181,119,56).  visible(lb1gt_0,56). happensAt(walking(lb1gt_0),56). orientation(lb1gt_1,147,56). coords(lb1gt_1,73,99,56).  visible(lb1gt_1,56). happensAt(walking(lb1gt_1),56).
orientation(lb1gt_0,148,57). coords(lb1gt_0,181,119,57).  visible(lb1gt_0,57). happensAt(active(lb1gt_0),57). orientation(lb1gt_1,147,57). coords(lb1gt_1,73,100,57).  visible(lb1gt_1,57). happensAt(walking(lb1gt_1),57).
orientation(lb1gt_0,148,58). coords(lb1gt_0,181,119,58).  visible(lb1gt_0,58). happensAt(active(lb1gt_0),58). orientation(lb1gt_1,147,58). coords(lb1gt_1,73,100,58).  visible(lb1gt_1,58). happensAt(walking(lb1gt_1),58).
orientation(lb1gt_0,148,59). coords(lb1gt_0,181,119,59).  visible(lb1gt_0,59). happensAt(active(lb1gt_0),59). orientation(lb1gt_1,147,59). coords(lb1gt_1,73,100,59).  visible(lb1gt_1,59). happensAt(walking(lb1gt_1),59).
orientation(lb1gt_0,148,60). coords(lb1gt_0,181,119,60).  visible(lb1gt_0,60). happensAt(active(lb1gt_0),60). orientation(lb1gt_1,147,60). coords(lb1gt_1,74,101,60).  visible(lb1gt_1,60). happensAt(walking(lb1gt_1),60).
orientation(lb1gt_0,148,61). coords(lb1gt_0,180,119,61).  visible(lb1gt_0,61). happensAt(active(lb1gt_0),61). orientation(lb1gt_1,147,61). coords(lb1gt_1,74,101,61).  visible(lb1gt_1,61). happensAt(walking(lb1gt_1),61).
orientation(lb1gt_0,148,62). coords(lb1gt_0,180,119,62).  visible(lb1gt_0,62). happensAt(active(lb1gt_0),62). orientation(lb1gt_1,147,62). coords(lb1gt_1,74,102,62).  visible(lb1gt_1,62). happensAt(walking(lb1gt_1),62).
orientation(lb1gt_0,148,63). coords(lb1gt_0,179,119,63).  visible(lb1gt_0,63). happensAt(active(lb1gt_0),63). orientation(lb1gt_1,147,63). coords(lb1gt_1,74,102,63).  visible(lb1gt_1,63). happensAt(walking(lb1gt_1),63).
orientation(lb1gt_0,148,64). coords(lb1gt_0,178,119,64).  visible(lb1gt_0,64). happensAt(active(lb1gt_0),64). orientation(lb1gt_1,147,64). coords(lb1gt_1,74,102,64).  visible(lb1gt_1,64). happensAt(walking(lb1gt_1),64).
orientation(lb1gt_0,148,65). coords(lb1gt_0,177,118,65).  visible(lb1gt_0,65). happensAt(active(lb1gt_0),65). orientation(lb1gt_1,147,65). coords(lb1gt_1,74,102,65).  visible(lb1gt_1,65). happensAt(walking(lb1gt_1),65).
orientation(lb1gt_0,148,66). coords(lb1gt_0,176,118,66).  visible(lb1gt_0,66). happensAt(active(lb1gt_0),66). orientation(lb1gt_1,147,66). coords(lb1gt_1,74,102,66).  visible(lb1gt_1,66). happensAt(walking(lb1gt_1),66).
orientation(lb1gt_0,148,67). coords(lb1gt_0,176,117,67).  visible(lb1gt_0,67). happensAt(active(lb1gt_0),67). orientation(lb1gt_1,147,67). coords(lb1gt_1,74,103,67).  visible(lb1gt_1,67). happensAt(walking(lb1gt_1),67).
orientation(lb1gt_0,148,68). coords(lb1gt_0,176,118,68).  visible(lb1gt_0,68). happensAt(active(lb1gt_0),68). orientation(lb1gt_1,147,68). coords(lb1gt_1,74,103,68).  visible(lb1gt_1,68). happensAt(walking(lb1gt_1),68).
orientation(lb1gt_0,148,69). coords(lb1gt_0,176,117,69).  visible(lb1gt_0,69). happensAt(active(lb1gt_0),69). orientation(lb1gt_1,147,69). coords(lb1gt_1,73,103,69).  visible(lb1gt_1,69). happensAt(walking(lb1gt_1),69).
orientation(lb1gt_0,148,70). coords(lb1gt_0,175,116,70).  visible(lb1gt_0,70). happensAt(active(lb1gt_0),70). orientation(lb1gt_1,147,70). coords(lb1gt_1,73,103,70).  visible(lb1gt_1,70). happensAt(walking(lb1gt_1),70).
orientation(lb1gt_0,148,71). coords(lb1gt_0,175,115,71).  visible(lb1gt_0,71). happensAt(active(lb1gt_0),71). orientation(lb1gt_1,147,71). coords(lb1gt_1,74,104,71).  visible(lb1gt_1,71). happensAt(walking(lb1gt_1),71).
orientation(lb1gt_0,148,72). coords(lb1gt_0,175,113,72).  visible(lb1gt_0,72). happensAt(active(lb1gt_0),72). orientation(lb1gt_1,147,72). coords(lb1gt_1,74,104,72).  visible(lb1gt_1,72). happensAt(walking(lb1gt_1),72).
#show holdsAt/2.