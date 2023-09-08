
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

	orientation(lb1gt_1,147,48). coords(lb1gt_1,73,92,48).  visible(lb1gt_1,48). happensAt(walking(lb1gt_1),48). orientation(lb1gt_2,142,48). coords(lb1gt_2,82,76,48).  visible(lb1gt_2,48). happensAt(walking(lb1gt_2),48).
orientation(lb1gt_1,147,49). coords(lb1gt_1,73,92,49).  visible(lb1gt_1,49). happensAt(walking(lb1gt_1),49). orientation(lb1gt_2,142,49). coords(lb1gt_2,82,76,49).  visible(lb1gt_2,49). happensAt(walking(lb1gt_2),49).
orientation(lb1gt_1,147,50). coords(lb1gt_1,73,92,50).  visible(lb1gt_1,50). happensAt(walking(lb1gt_1),50). orientation(lb1gt_2,142,50). coords(lb1gt_2,82,76,50).  visible(lb1gt_2,50). happensAt(walking(lb1gt_2),50).
orientation(lb1gt_1,147,51). coords(lb1gt_1,73,93,51).  visible(lb1gt_1,51). happensAt(walking(lb1gt_1),51). orientation(lb1gt_2,142,51). coords(lb1gt_2,82,77,51).  visible(lb1gt_2,51). happensAt(walking(lb1gt_2),51).
orientation(lb1gt_1,147,52). coords(lb1gt_1,73,94,52).  visible(lb1gt_1,52). happensAt(walking(lb1gt_1),52). orientation(lb1gt_2,142,52). coords(lb1gt_2,82,78,52).  visible(lb1gt_2,52). happensAt(walking(lb1gt_2),52).
orientation(lb1gt_1,147,53). coords(lb1gt_1,73,95,53).  visible(lb1gt_1,53). happensAt(walking(lb1gt_1),53). orientation(lb1gt_2,142,53). coords(lb1gt_2,82,78,53).  visible(lb1gt_2,53). happensAt(walking(lb1gt_2),53).
orientation(lb1gt_1,147,54). coords(lb1gt_1,73,97,54).  visible(lb1gt_1,54). happensAt(walking(lb1gt_1),54). orientation(lb1gt_2,142,54). coords(lb1gt_2,82,78,54).  visible(lb1gt_2,54). happensAt(walking(lb1gt_2),54).
orientation(lb1gt_1,147,55). coords(lb1gt_1,73,98,55).  visible(lb1gt_1,55). happensAt(walking(lb1gt_1),55). orientation(lb1gt_2,142,55). coords(lb1gt_2,82,78,55).  visible(lb1gt_2,55). happensAt(walking(lb1gt_2),55).
orientation(lb1gt_1,147,56). coords(lb1gt_1,73,99,56).  visible(lb1gt_1,56). happensAt(walking(lb1gt_1),56). orientation(lb1gt_2,142,56). coords(lb1gt_2,82,79,56).  visible(lb1gt_2,56). happensAt(walking(lb1gt_2),56).
orientation(lb1gt_1,147,57). coords(lb1gt_1,73,100,57).  visible(lb1gt_1,57). happensAt(walking(lb1gt_1),57). orientation(lb1gt_2,142,57). coords(lb1gt_2,83,79,57).  visible(lb1gt_2,57). happensAt(walking(lb1gt_2),57).
orientation(lb1gt_1,147,58). coords(lb1gt_1,73,100,58).  visible(lb1gt_1,58). happensAt(walking(lb1gt_1),58). orientation(lb1gt_2,142,58). coords(lb1gt_2,83,79,58).  visible(lb1gt_2,58). happensAt(walking(lb1gt_2),58).
orientation(lb1gt_1,147,59). coords(lb1gt_1,73,100,59).  visible(lb1gt_1,59). happensAt(walking(lb1gt_1),59). orientation(lb1gt_2,142,59). coords(lb1gt_2,83,79,59).  visible(lb1gt_2,59). happensAt(walking(lb1gt_2),59).
orientation(lb1gt_1,147,60). coords(lb1gt_1,74,101,60).  visible(lb1gt_1,60). happensAt(walking(lb1gt_1),60). orientation(lb1gt_2,142,60). coords(lb1gt_2,83,79,60).  visible(lb1gt_2,60). happensAt(walking(lb1gt_2),60).
orientation(lb1gt_1,147,61). coords(lb1gt_1,74,101,61).  visible(lb1gt_1,61). happensAt(walking(lb1gt_1),61). orientation(lb1gt_2,142,61). coords(lb1gt_2,83,80,61).  visible(lb1gt_2,61). happensAt(walking(lb1gt_2),61).
orientation(lb1gt_1,147,62). coords(lb1gt_1,74,102,62).  visible(lb1gt_1,62). happensAt(walking(lb1gt_1),62). orientation(lb1gt_2,142,62). coords(lb1gt_2,83,80,62).  visible(lb1gt_2,62). happensAt(walking(lb1gt_2),62).
orientation(lb1gt_1,147,63). coords(lb1gt_1,74,102,63).  visible(lb1gt_1,63). happensAt(walking(lb1gt_1),63). orientation(lb1gt_2,142,63). coords(lb1gt_2,83,80,63).  visible(lb1gt_2,63). happensAt(walking(lb1gt_2),63).
orientation(lb1gt_1,147,64). coords(lb1gt_1,74,102,64).  visible(lb1gt_1,64). happensAt(walking(lb1gt_1),64). orientation(lb1gt_2,142,64). coords(lb1gt_2,83,80,64).  visible(lb1gt_2,64). happensAt(walking(lb1gt_2),64).
orientation(lb1gt_1,147,65). coords(lb1gt_1,74,102,65).  visible(lb1gt_1,65). happensAt(walking(lb1gt_1),65). orientation(lb1gt_2,142,65). coords(lb1gt_2,84,80,65).  visible(lb1gt_2,65). happensAt(walking(lb1gt_2),65).
orientation(lb1gt_1,147,66). coords(lb1gt_1,74,102,66).  visible(lb1gt_1,66). happensAt(walking(lb1gt_1),66). orientation(lb1gt_2,142,66). coords(lb1gt_2,84,81,66).  visible(lb1gt_2,66). happensAt(walking(lb1gt_2),66).
orientation(lb1gt_1,147,67). coords(lb1gt_1,74,103,67).  visible(lb1gt_1,67). happensAt(walking(lb1gt_1),67). orientation(lb1gt_2,142,67). coords(lb1gt_2,84,81,67).  visible(lb1gt_2,67). happensAt(walking(lb1gt_2),67).
orientation(lb1gt_1,147,68). coords(lb1gt_1,74,103,68).  visible(lb1gt_1,68). happensAt(walking(lb1gt_1),68). orientation(lb1gt_2,142,68). coords(lb1gt_2,85,81,68).  visible(lb1gt_2,68). happensAt(walking(lb1gt_2),68).
orientation(lb1gt_1,147,69). coords(lb1gt_1,73,103,69).  visible(lb1gt_1,69). happensAt(walking(lb1gt_1),69). orientation(lb1gt_2,142,69). coords(lb1gt_2,85,81,69).  visible(lb1gt_2,69). happensAt(walking(lb1gt_2),69).
orientation(lb1gt_1,147,70). coords(lb1gt_1,73,103,70).  visible(lb1gt_1,70). happensAt(walking(lb1gt_1),70). orientation(lb1gt_2,142,70). coords(lb1gt_2,86,81,70).  visible(lb1gt_2,70). happensAt(walking(lb1gt_2),70).
orientation(lb1gt_1,147,71). coords(lb1gt_1,74,104,71).  visible(lb1gt_1,71). happensAt(walking(lb1gt_1),71). orientation(lb1gt_2,142,71). coords(lb1gt_2,86,81,71).  visible(lb1gt_2,71). happensAt(walking(lb1gt_2),71).
orientation(lb1gt_1,147,72). coords(lb1gt_1,74,104,72).  visible(lb1gt_1,72). happensAt(walking(lb1gt_1),72). orientation(lb1gt_2,142,72). coords(lb1gt_2,87,81,72).  visible(lb1gt_2,72). happensAt(walking(lb1gt_2),72).
#show holdsAt/2.