
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

	orientation(rsfgt_0,160,48). coords(rsfgt_0,90,134,48).  visible(rsfgt_0,48). happensAt(walking(rsfgt_0),48). orientation(rsfgt_1,150,48). coords(rsfgt_1,183,134,48).  visible(rsfgt_1,48). happensAt(walking(rsfgt_1),48).
orientation(rsfgt_0,160,49). coords(rsfgt_0,90,134,49).  visible(rsfgt_0,49). happensAt(walking(rsfgt_0),49). orientation(rsfgt_1,150,49). coords(rsfgt_1,183,133,49).  visible(rsfgt_1,49). happensAt(walking(rsfgt_1),49).
orientation(rsfgt_0,160,50). coords(rsfgt_0,89,134,50).  visible(rsfgt_0,50). happensAt(walking(rsfgt_0),50). orientation(rsfgt_1,150,50). coords(rsfgt_1,184,133,50).  visible(rsfgt_1,50). happensAt(walking(rsfgt_1),50).
orientation(rsfgt_0,160,51). coords(rsfgt_0,87,134,51).  visible(rsfgt_0,51). happensAt(walking(rsfgt_0),51). orientation(rsfgt_1,150,51). coords(rsfgt_1,184,131,51).  visible(rsfgt_1,51). happensAt(walking(rsfgt_1),51).
orientation(rsfgt_0,160,52). coords(rsfgt_0,85,134,52).  visible(rsfgt_0,52). happensAt(walking(rsfgt_0),52). orientation(rsfgt_1,150,52). coords(rsfgt_1,183,128,52).  visible(rsfgt_1,52). happensAt(walking(rsfgt_1),52).
orientation(rsfgt_0,160,53). coords(rsfgt_0,82,134,53).  visible(rsfgt_0,53). happensAt(walking(rsfgt_0),53). orientation(rsfgt_1,150,53). coords(rsfgt_1,183,127,53).  visible(rsfgt_1,53). happensAt(walking(rsfgt_1),53).
orientation(rsfgt_0,160,54). coords(rsfgt_0,82,134,54).  visible(rsfgt_0,54). happensAt(walking(rsfgt_0),54). orientation(rsfgt_1,150,54). coords(rsfgt_1,182,127,54).  visible(rsfgt_1,54). happensAt(walking(rsfgt_1),54).
orientation(rsfgt_0,160,55). coords(rsfgt_0,81,134,55).  visible(rsfgt_0,55). happensAt(walking(rsfgt_0),55). orientation(rsfgt_1,150,55). coords(rsfgt_1,179,126,55).  visible(rsfgt_1,55). happensAt(walking(rsfgt_1),55).
orientation(rsfgt_0,160,56). coords(rsfgt_0,80,134,56).  visible(rsfgt_0,56). happensAt(walking(rsfgt_0),56). orientation(rsfgt_1,150,56). coords(rsfgt_1,178,126,56).  visible(rsfgt_1,56). happensAt(walking(rsfgt_1),56).
orientation(rsfgt_0,160,57). coords(rsfgt_0,80,134,57).  visible(rsfgt_0,57). happensAt(walking(rsfgt_0),57). orientation(rsfgt_1,150,57). coords(rsfgt_1,178,123,57).  visible(rsfgt_1,57). happensAt(walking(rsfgt_1),57).
orientation(rsfgt_0,160,58). coords(rsfgt_0,80,134,58).  visible(rsfgt_0,58). happensAt(walking(rsfgt_0),58). orientation(rsfgt_1,150,58). coords(rsfgt_1,178,122,58).  visible(rsfgt_1,58). happensAt(walking(rsfgt_1),58).
orientation(rsfgt_0,160,59). coords(rsfgt_0,80,136,59).  visible(rsfgt_0,59). happensAt(walking(rsfgt_0),59). orientation(rsfgt_1,150,59). coords(rsfgt_1,177,121,59).  visible(rsfgt_1,59). happensAt(walking(rsfgt_1),59).
orientation(rsfgt_0,160,60). coords(rsfgt_0,79,136,60).  visible(rsfgt_0,60). happensAt(walking(rsfgt_0),60). orientation(rsfgt_1,150,60). coords(rsfgt_1,177,117,60).  visible(rsfgt_1,60). happensAt(walking(rsfgt_1),60).
orientation(rsfgt_0,160,61). coords(rsfgt_0,78,137,61).  visible(rsfgt_0,61). happensAt(walking(rsfgt_0),61). orientation(rsfgt_1,150,61). coords(rsfgt_1,176,117,61).  visible(rsfgt_1,61). happensAt(walking(rsfgt_1),61).
orientation(rsfgt_0,160,62). coords(rsfgt_0,78,137,62).  visible(rsfgt_0,62). happensAt(walking(rsfgt_0),62). orientation(rsfgt_1,150,62). coords(rsfgt_1,175,117,62).  visible(rsfgt_1,62). happensAt(walking(rsfgt_1),62).
orientation(rsfgt_0,160,63). coords(rsfgt_0,78,137,63).  visible(rsfgt_0,63). happensAt(walking(rsfgt_0),63). orientation(rsfgt_1,150,63). coords(rsfgt_1,174,117,63).  visible(rsfgt_1,63). happensAt(walking(rsfgt_1),63).
orientation(rsfgt_0,160,64). coords(rsfgt_0,75,136,64).  visible(rsfgt_0,64). happensAt(walking(rsfgt_0),64). orientation(rsfgt_1,150,64). coords(rsfgt_1,174,117,64).  visible(rsfgt_1,64). happensAt(walking(rsfgt_1),64).
orientation(rsfgt_0,160,65). coords(rsfgt_0,73,137,65).  visible(rsfgt_0,65). happensAt(walking(rsfgt_0),65). orientation(rsfgt_1,150,65). coords(rsfgt_1,174,117,65).  visible(rsfgt_1,65). happensAt(walking(rsfgt_1),65).
orientation(rsfgt_0,160,66). coords(rsfgt_0,72,138,66).  visible(rsfgt_0,66). happensAt(walking(rsfgt_0),66). orientation(rsfgt_1,150,66). coords(rsfgt_1,174,117,66).  visible(rsfgt_1,66). happensAt(walking(rsfgt_1),66).
orientation(rsfgt_0,160,67). coords(rsfgt_0,71,139,67).  visible(rsfgt_0,67). happensAt(walking(rsfgt_0),67). orientation(rsfgt_1,150,67). coords(rsfgt_1,174,117,67).  visible(rsfgt_1,67). happensAt(walking(rsfgt_1),67).
orientation(rsfgt_0,160,68). coords(rsfgt_0,71,139,68).  visible(rsfgt_0,68). happensAt(walking(rsfgt_0),68). orientation(rsfgt_1,150,68). coords(rsfgt_1,173,117,68).  visible(rsfgt_1,68). happensAt(walking(rsfgt_1),68).
orientation(rsfgt_0,160,69). coords(rsfgt_0,71,139,69).  visible(rsfgt_0,69). happensAt(walking(rsfgt_0),69). orientation(rsfgt_1,150,69). coords(rsfgt_1,173,118,69).  visible(rsfgt_1,69). happensAt(walking(rsfgt_1),69).
orientation(rsfgt_0,160,70). coords(rsfgt_0,71,139,70).  visible(rsfgt_0,70). happensAt(walking(rsfgt_0),70). orientation(rsfgt_1,150,70). coords(rsfgt_1,172,117,70).  visible(rsfgt_1,70). happensAt(walking(rsfgt_1),70).
orientation(rsfgt_0,160,71). coords(rsfgt_0,71,140,71).  visible(rsfgt_0,71). happensAt(walking(rsfgt_0),71). orientation(rsfgt_1,150,71). coords(rsfgt_1,170,116,71).  visible(rsfgt_1,71). happensAt(walking(rsfgt_1),71).
orientation(rsfgt_0,160,72). coords(rsfgt_0,70,141,72).  visible(rsfgt_0,72). happensAt(walking(rsfgt_0),72). orientation(rsfgt_1,150,72). coords(rsfgt_1,170,116,72).  visible(rsfgt_1,72). happensAt(walking(rsfgt_1),72).
#show holdsAt/2.