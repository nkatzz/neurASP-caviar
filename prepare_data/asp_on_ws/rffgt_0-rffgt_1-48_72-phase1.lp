
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

	orientation(rffgt_0,137,48). coords(rffgt_0,138,63,48).  visible(rffgt_0,48). happensAt(active(rffgt_0),48). orientation(rffgt_1,173,48). coords(rffgt_1,161,106,48).  visible(rffgt_1,48). happensAt(active(rffgt_1),48).
orientation(rffgt_0,137,49). coords(rffgt_0,138,63,49).  visible(rffgt_0,49). happensAt(active(rffgt_0),49). orientation(rffgt_1,173,49). coords(rffgt_1,161,106,49).  visible(rffgt_1,49). happensAt(active(rffgt_1),49).
orientation(rffgt_0,137,50). coords(rffgt_0,138,62,50).  visible(rffgt_0,50). happensAt(active(rffgt_0),50). orientation(rffgt_1,173,50). coords(rffgt_1,161,106,50).  visible(rffgt_1,50). happensAt(active(rffgt_1),50).
orientation(rffgt_0,137,51). coords(rffgt_0,138,62,51).  visible(rffgt_0,51). happensAt(active(rffgt_0),51). orientation(rffgt_1,171,51). coords(rffgt_1,161,106,51).  visible(rffgt_1,51). happensAt(active(rffgt_1),51).
orientation(rffgt_0,137,52). coords(rffgt_0,138,62,52).  visible(rffgt_0,52). happensAt(active(rffgt_0),52). orientation(rffgt_1,171,52). coords(rffgt_1,161,106,52).  visible(rffgt_1,52). happensAt(active(rffgt_1),52).
orientation(rffgt_0,137,53). coords(rffgt_0,136,61,53).  visible(rffgt_0,53). happensAt(active(rffgt_0),53). orientation(rffgt_1,167,53). coords(rffgt_1,161,106,53).  visible(rffgt_1,53). happensAt(active(rffgt_1),53).
orientation(rffgt_0,137,54). coords(rffgt_0,136,61,54).  visible(rffgt_0,54). happensAt(active(rffgt_0),54). orientation(rffgt_1,165,54). coords(rffgt_1,161,106,54).  visible(rffgt_1,54). happensAt(active(rffgt_1),54).
orientation(rffgt_0,137,55). coords(rffgt_0,136,61,55).  visible(rffgt_0,55). happensAt(active(rffgt_0),55). orientation(rffgt_1,161,55). coords(rffgt_1,160,105,55).  visible(rffgt_1,55). happensAt(active(rffgt_1),55).
orientation(rffgt_0,137,56). coords(rffgt_0,136,61,56).  visible(rffgt_0,56). happensAt(active(rffgt_0),56). orientation(rffgt_1,161,56). coords(rffgt_1,160,105,56).  visible(rffgt_1,56). happensAt(active(rffgt_1),56).
orientation(rffgt_0,137,57). coords(rffgt_0,136,61,57).  visible(rffgt_0,57). happensAt(active(rffgt_0),57). orientation(rffgt_1,158,57). coords(rffgt_1,160,105,57).  visible(rffgt_1,57). happensAt(active(rffgt_1),57).
orientation(rffgt_0,137,58). coords(rffgt_0,136,61,58).  visible(rffgt_0,58). happensAt(active(rffgt_0),58). orientation(rffgt_1,155,58). coords(rffgt_1,159,106,58).  visible(rffgt_1,58). happensAt(active(rffgt_1),58).
orientation(rffgt_0,137,59). coords(rffgt_0,136,61,59).  visible(rffgt_0,59). happensAt(active(rffgt_0),59). orientation(rffgt_1,155,59). coords(rffgt_1,158,107,59).  visible(rffgt_1,59). happensAt(active(rffgt_1),59).
orientation(rffgt_0,137,60). coords(rffgt_0,136,61,60).  visible(rffgt_0,60). happensAt(active(rffgt_0),60). orientation(rffgt_1,148,60). coords(rffgt_1,157,108,60).  visible(rffgt_1,60). happensAt(active(rffgt_1),60).
orientation(rffgt_0,137,61). coords(rffgt_0,136,61,61).  visible(rffgt_0,61). happensAt(active(rffgt_0),61). orientation(rffgt_1,148,61). coords(rffgt_1,155,109,61).  visible(rffgt_1,61). happensAt(active(rffgt_1),61).
orientation(rffgt_0,137,62). coords(rffgt_0,136,62,62).  visible(rffgt_0,62). happensAt(active(rffgt_0),62). orientation(rffgt_1,144,62). coords(rffgt_1,155,111,62).  visible(rffgt_1,62). happensAt(active(rffgt_1),62).
orientation(rffgt_0,137,63). coords(rffgt_0,136,62,63).  visible(rffgt_0,63). happensAt(active(rffgt_0),63). orientation(rffgt_1,144,63). coords(rffgt_1,155,112,63).  visible(rffgt_1,63). happensAt(active(rffgt_1),63).
orientation(rffgt_0,137,64). coords(rffgt_0,136,62,64).  visible(rffgt_0,64). happensAt(active(rffgt_0),64). orientation(rffgt_1,144,64). coords(rffgt_1,154,112,64).  visible(rffgt_1,64). happensAt(active(rffgt_1),64).
orientation(rffgt_0,137,65). coords(rffgt_0,136,62,65).  visible(rffgt_0,65). happensAt(active(rffgt_0),65). orientation(rffgt_1,139,65). coords(rffgt_1,155,112,65).  visible(rffgt_1,65). happensAt(active(rffgt_1),65).
orientation(rffgt_0,137,66). coords(rffgt_0,136,61,66).  visible(rffgt_0,66). happensAt(active(rffgt_0),66). orientation(rffgt_1,139,66). coords(rffgt_1,155,113,66).  visible(rffgt_1,66). happensAt(active(rffgt_1),66).
orientation(rffgt_0,137,67). coords(rffgt_0,136,61,67).  visible(rffgt_0,67). happensAt(active(rffgt_0),67). orientation(rffgt_1,139,67). coords(rffgt_1,155,113,67).  visible(rffgt_1,67). happensAt(active(rffgt_1),67).
orientation(rffgt_0,137,68). coords(rffgt_0,136,61,68).  visible(rffgt_0,68). happensAt(active(rffgt_0),68). orientation(rffgt_1,136,68). coords(rffgt_1,155,113,68).  visible(rffgt_1,68). happensAt(active(rffgt_1),68).
orientation(rffgt_0,137,69). coords(rffgt_0,135,61,69).  visible(rffgt_0,69). happensAt(active(rffgt_0),69). orientation(rffgt_1,136,69). coords(rffgt_1,155,113,69).  visible(rffgt_1,69). happensAt(active(rffgt_1),69).
orientation(rffgt_0,137,70). coords(rffgt_0,134,61,70).  visible(rffgt_0,70). happensAt(active(rffgt_0),70). orientation(rffgt_1,135,70). coords(rffgt_1,155,113,70).  visible(rffgt_1,70). happensAt(active(rffgt_1),70).
orientation(rffgt_0,134,71). coords(rffgt_0,134,61,71).  visible(rffgt_0,71). happensAt(active(rffgt_0),71). orientation(rffgt_1,134,71). coords(rffgt_1,155,113,71).  visible(rffgt_1,71). happensAt(active(rffgt_1),71).
orientation(rffgt_0,134,72). coords(rffgt_0,134,61,72).  visible(rffgt_0,72). happensAt(active(rffgt_0),72). orientation(rffgt_1,134,72). coords(rffgt_1,155,113,72).  visible(rffgt_1,72). happensAt(active(rffgt_1),72).
#show holdsAt/2.